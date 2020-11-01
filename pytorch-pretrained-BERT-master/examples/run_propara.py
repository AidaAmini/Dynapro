# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  GlobalQuestionAnsweringWithLocationalPredictions,
                                  Sequence_decoder,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer, RobertaConfig,
                                  RobertaTokenizer,
                                  DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
                                    
from pytorch_pretrained_bert.transformers import AdamW, WarmupLinearSchedule

from utils_propara import (read_propara_examples, convert_examples_to_features, RawResult, write_predictions)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
# from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'dygapro': (BertConfig, GlobalQuestionAnsweringWithLocationalPredictions, BertTokenizer),   ##
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
}

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, decoder, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=None, batch_size=args.train_batch_size) #changed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = []
    if model != None:
      param_optimizer = list(model.named_parameters())
    if decoder != None:
      param_optimizer_decoder = list(decoder.named_parameters())
      for param in param_optimizer_decoder:
        param_optimizer.append(param)

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]] #comparision

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  # optimizer is different than before
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=False)  # optimizer is different than before
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)  # the model didn't have schedular before
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        if decoder != None:
            decoder = torch.nn.DataParallel(decoder)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        if decoder !=  None:    
            decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    if decoder != None: 
        decoder.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    ave_loss = 0
    loss_sum = 0
    loss  = torch.zeros(1).cuda()
    for n_iter in train_iterator:
        print (ave_loss)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        loss_sum = 0
        for step, batch in enumerate(epoch_iterator):
            if step%50 ==0:
              print ("ave_loss + "  +str(ave_loss))
            if model != None:
                model.train()
            if decoder != None:
                decoder.train()

            train_iterator.set_description("Loss %f" % ave_loss)

            batch = tuple(t.to(args.device) for t in batch)
            
            #replaced old way not the dictionary way.
            input_ids, input_mask, \
                segment_ids, start_positions, \
                before_state_start_position, end_positions, \
                before_state_end_position, answer_mask, before_state_answer_mask, \
                known_switch, before_state_known_switch, actions, unk_mask, \
                none_mask, ans_mask = batch

            loss  = torch.zeros(1).cuda()
            loss_1 = torch.zeros(1).cuda()
            
            if model != None:
                loss, start_probs, end_probs, switch_probs, before_state_start_probs, \
                    before_state_end_probs, before_state_switch_probs, sequence_output, action_probs = \
                    model(input_ids, segment_ids, input_mask, \
                    start_positions, end_positions, answer_mask, known_switch, \
                    before_state_start_positions=before_state_start_position, \
                    before_state_end_positions=before_state_end_position, \
                    before_state_answer_mask=before_state_answer_mask, \
                    before_state_known_switch=before_state_known_switch, \
                    weight_class_loss=args.weight_class, weight_span_loss =args.weight_span, actions=actions)

            if decoder != None:
                loss_1 = decoder(input_ids, sequence_output, start_probs, end_probs, switch_probs, before_state_start_probs, before_state_end_probs, before_state_switch_probs, actions, unk_mask, none_mask, ans_mask)
         
            loss = loss + loss_1 * args.weight_actions
            
            loss_sum = loss[0].data.cpu().numpy() + loss_sum
            ave_loss = loss_sum / step

            # outputs = model(**inputs)
            # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)


            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if decoder != None:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                #this update of lr is happening in the previous code I added here
                # lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                # for param_group in optimizer.param_groups:
                #         param_group['lr'] = lr_this_step
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                if decoder != None:
                    decoder.zero_grad()
                global_step += 1


                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if I define a good evaluate metric here I can have the good diagram here.
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Here they save the model every s steps I am changing it to be at the end of every epoch same as before
                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        if n_iter > 5 or args.train_from_checkpoint:
          if decoder != None:
            decoder_to_save = decoder.module if hasattr(decoder, 'module') else decoder  # Only save the global_model it-self
            output_decoder_file = os.path.join(args.output_dir, "pytorch_model_decoder"+str(n_iter)+".bin")
          if model != None:
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the global_model it-self
            output_model_file = os.path.join(args.output_dir, "pytorch_model"+str(n_iter)+".bin")
          if args.train_from_checkpoint:
            if decoder != None:
                decoder_to_save = decoder.module if hasattr(decoder, 'module') else decoder  # Only save the global_model it-self
                output_decoder_file = os.path.join(args.output_dir, "pytorch_model_decoder"+str(n_iter+49)+".bin")
            if model != None:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the global_model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model"+str(n_iter+49)+".bin")
          if args.do_train:
            if model != None:
              torch.save(model_to_save.state_dict(), output_model_file)
            if decoder != None:
              torch.save(decoder_to_save.state_dict(), output_decoder_file)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, decoder, tokenizer, prefix="", dataset=None, examples=None, features=None):

    if dataset == None:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=None, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if model != None:
            model.eval()
        if decoder != None:
            decoder.eval()
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, example_indices, unk_mask, none_mask, ans_mask = batch
        
        # batch_start_logits, batch_end_logits, batch_switch_logits, batch_before_state_start_logits, batch_before_state_end_logits, batch_before_state_switch_logits, sequence_output, batch_action_logits =  None
        if model != None:
            batch_start_logits, batch_end_logits, batch_switch_logits, batch_before_state_start_logits, batch_before_state_end_logits, batch_before_state_switch_logits, sequence_output, batch_action_logits =  model(input_ids, segment_ids, input_mask)
        if decoder != None:
            batch_action_logits = decoder(input_ids, sequence_output, batch_start_logits, batch_end_logits, batch_switch_logits, batch_before_state_start_logits, batch_before_state_end_logits, batch_before_state_switch_logits, None, unk_mask, none_mask, ans_mask)

        # with torch.no_grad():
        #     inputs = {'input_ids':      batch[0],
        #               'attention_mask': batch[1],
        #               'token_type_ids': None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
        #               }
        #     example_indices = batch[3]
        #     if args.model_type in ['xlnet', 'xlm']:
        #         inputs.update({'cls_index': batch[4],
        #                        'p_mask':    batch[5]})
        #     outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            for j in range(1):
                if sum(input_ids[i][j]).cpu().numpy() == 0:
                      continue
                
                start_logits = [0 for x in range(200)]
                end_logits = [0 for x in range(200)]
                switch_logits = [0 for x in range(3)]
                before_state_start_logits = [0 for x in range(200)]
                before_state_end_logits = [0 for x in range(200)]
                before_state_switch_logits = [0 for x in range(3)]
                action_logits = [0 for x in range(4)]

                if batch_action_logits != []:
                    action_logits = batch_action_logits[j][i].detach().cpu().tolist()

                if batch_start_logits != []:
                    start_logits = batch_start_logits[j][i].detach().cpu().tolist()
                if batch_end_logits != []:
                    end_logits = batch_end_logits[j][i].detach().cpu().tolist()  
                if batch_switch_logits != []:
                    switch_logits = batch_switch_logits[j][i].detach().cpu().tolist()
                if batch_before_state_start_logits != []:
                    before_state_start_logits = batch_before_state_start_logits[j][i].detach().cpu().tolist()
                if batch_before_state_end_logits != []:
                    before_state_end_logits = batch_before_state_end_logits[j][i].detach().cpu().tolist()
                if batch_before_state_switch_logits != []:
                    before_state_switch_logits = batch_before_state_switch_logits[j][i].detach().cpu().tolist()

                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id_list[j])
                all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits, 
                                                 switch_logits=switch_logits, 
                                                 before_state_start_logits=before_state_start_logits, 
                                                 before_state_end_logits=before_state_end_logits, 
                                                 before_state_switch_logits=before_state_switch_logits, 
                                                 action_logits=action_logits))

    # Compute predictions
    # output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    # output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if not os.path.exists(os.path.join(args.output_dir, 'stats')):
        os.makedirs(os.path.join(args.output_dir, 'stats'))
    
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}_".format(prefix) + "_.json")
    output_prediction_file_before_state = os.path.join(args.output_dir, "b_predictions_{}_".format(prefix) + "_.json")
    output_prediction_file_actions = os.path.join(args.output_dir, "action_predictions_{}_".format(prefix) + "_.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_".format(prefix) + "_.json")
    output_stat_path_file = os.path.join(os.path.join(args.output_dir, 'stats'), "stats_{}_".format(prefix) + ".txt")
    
    write_predictions(examples, features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,output_prediction_file_before_state,output_prediction_file_actions,
                      output_nbest_file, args.verbose_logging, tokenizer=tokenizer)
    

    # if args.model_type in ['xlnet', 'xlm']:
    #     # XLNet uses a more complex post-processing procedure
    #     write_predictions_extended(examples, features, all_results, args.n_best_size,
    #                     args.max_answer_length, output_prediction_file,
    #                     output_nbest_file, output_null_log_odds_file, args.predict_file,
    #                     model.config.start_n_top, model.config.end_n_top,
    #                     args.version_2_with_negative, tokenizer, args.verbose_logging)

    # write_predictions(examples, features, all_results, args.n_best_size,
    #                     args.max_answer_length, args.do_lower_case, output_prediction_file,
    #                     output_nbest_file, output_null_log_odds_file, args.verbose_logging,
    #                     args.version_2_with_negative, args.null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    # evaluate_options = EVAL_OPTS(data_file=args.predict_file,
    #                              pred_file=output_prediction_file,
    #                              na_prob_file=output_null_log_odds_file)
    # results = evaluate_on_squad(evaluate_options)
    return 'None'


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_propara_examples(input_file=input_file,
                                                is_training=not evaluate)
        
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate,
                                                do_full_context=args.do_full_context)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    if not evaluate:
        all_input_ids = torch.tensor([f.input_ids_list for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask_list for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask_list for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids_list for f in features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position_list for f in features], dtype=torch.long)
        all_before_state_start_positions = torch.tensor([f.before_state_start_position_list for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position_list for f in features], dtype=torch.long)
        all_before_state_end_positions = torch.tensor([f.before_state_end_position_list for f in features], dtype=torch.long)
        all_answer_masks = torch.tensor([f.answer_mask_list for f in features], dtype=torch.long)
        all_before_state_answer_masks = torch.tensor([f.before_state_answer_mask_list for f in features], dtype=torch.long)
        all_known_switchs = torch.tensor([f.known_switch_list for f in features], dtype=torch.long)
        all_before_state_known_switchs = torch.tensor([f.before_state_known_switch_list for f in features], dtype=torch.long)
        all_action_list = torch.tensor([f.action_list for f in features], dtype=torch.long)
        all_unk_mask_list = torch.tensor([f.unk_mask_list for f in features], dtype=torch.float)
        all_none_mask_list = torch.tensor([f.none_mask_list for f in features], dtype=torch.float)
        all_ans_mask_list = torch.tensor([f.ans_mask_list for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_before_state_start_positions, all_end_positions, all_before_state_end_positions,
                                   all_answer_masks, all_before_state_answer_masks, all_known_switchs, all_before_state_known_switchs, all_action_list,
                                   all_unk_mask_list, all_none_mask_list, all_ans_mask_list)
    else:
        all_input_ids = torch.tensor([f.input_ids_list for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask_list for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids_list for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_unk_mask_list = torch.tensor([f.unk_mask_list for f in features], dtype=torch.float)
        all_none_mask_list = torch.tensor([f.none_mask_list for f in features], dtype=torch.float)
        all_ans_mask_list = torch.tensor([f.ans_mask_list for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_unk_mask_list, all_none_mask_list, all_ans_mask_list)
           
    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=10, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_full_context", action='store_true',
                        help="Whether to pass the whole document level to the model or not!!!")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    
    # My code 
    parser.add_argument('--weight_class',
                        type=float, default=1,
                        help="weight of the loss on the class prediction\n")

    parser.add_argument('--weight_span',
                        type=float, default=1,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n")

    parser.add_argument('--weight_actions',
                        type=float, default=1,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n")

    parser.add_argument('--train_from_checkpoint', action='store_true', help="Whether to run training.")
    parser.add_argument("--trained_encoder_saved", default=None, type=str, help="Path to the saved encoder")
    parser.add_argument("--trained_decoder_saved", default=None, type=str, help="path to saved decoder")


    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    #newly code TODO double check if need upgrading
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    

    model, decoder = initialize_model_decoder(args, model_class, config)

    if args.local_rank == 0:
        torch.distributed.barrier() 

    if model != None:
        model.to(args.device)
    if decoder != None:
        decoder.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, decoder, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}

    if args.do_eval and args.local_rank in [-1, 0]:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
        for checkpoint in range(41, 42):
            # Reload the model
            if args.trained_encoder_saved != None:
              output_model_file = os.path.join(args.output_dir, args.trained_encoder_saved)
            else:
              output_model_file = os.path.join(args.output_dir, "pytorch_model"+str(checkpoint)+".bin")
            
            output_decoder_file = None
            if args.trained_decoder_saved != None:
              output_decoder_file = os.path.join(args.output_dir, args.trained_decoder_saved)
            else:  
              output_decoder_file = os.path.join(args.output_dir, "pytorch_model_decoder"+str(checkpoint)+".bin")
    

            model, decoder = load_pretrained_model_decoder(args, output_model_file, output_decoder_file, model_class)
            # model_class.from_pretrained(checkpoint)
            if model != None:
                model.to(args.device)
            if decoder != None:
                decoder.to(args.device)
            # Evaluate
            result = evaluate(args, model, decoder, tokenizer, prefix=str(checkpoint), dataset=dataset, examples=examples, features=features)

            # result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            # results.update(result)

    logger.info("Results: {}".format(results))

    return results

def load_pretrained_model_decoder(args, model_path, decoder_path, model_class):
    model = None
    decoder = None

    model_state_dict = torch.load(model_path)
    model  = model_class.from_pretrained(args.model_name_or_path, state_dict=model_state_dict)
    decoder = Sequence_decoder()
    decoder.load_state_dict(torch.load(decoder_path))   
    return model, decoder

def initialize_model_decoder(args, model_class, config):
    model = None
    decoder = None
    if args.train_from_checkpoint:
        #if not mentioned the default value is set for model and decoder file names.
        output_model_file = os.path.join(args.output_dir, "pytorch_model49.bin")
        output_decoder_file = os.path.join(args.output_dir, "pytorch_model_decoder49.bin")

        if args.trained_encoder_saved != None:
            output_model_file = os.path.join(args.output_dir, args.trained_encoder_saved)
        if args.trained_decoder_saved != None:
            output_decoder_file = os.path.join(args.output_dir, args.trained_decoder_saved)
        
        model_state_dict = torch.load(output_model_file)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        # model = GlobalQuestionAnsweringWithLocationalPredictions.from_pretrained(args.bert_model, state_dict=model_state_dict)
        decoder = Sequence_decoder()
        decoder.load_state_dict(torch.load(output_decoder_file))

    else:
        decoder = Sequence_decoder()
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    return model, decoder

if __name__ == "__main__":
    main()
