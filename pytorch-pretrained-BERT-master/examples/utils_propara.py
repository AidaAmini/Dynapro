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
""" BERT multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function

import collections
import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
import math

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer


max_num_of_events = 1
logger = logging.getLogger(__name__)


class ProparaExample(object):
    """A single training/test example for the Squad dataset."""

    def __init__(self,
                 qas_id,
                 global_paragraph,
                 question_text_list,
                 doc_tokens_list,
                 future_doc_tokens_list=None,
                 orig_answer_text_list=None,
                 before_state_answer_list=None,
                 start_position_list=None,
                 before_state_start_position_list=None,
                 end_position_list=None,
                 before_state_end_position_list=None,
                 answer_mask_list=None,
                 before_state_answer_mask_list=None,
                 known_switch_list=None,
                 before_state_known_switch_list=None,
                 action_list = None):
        
        self.qas_id = qas_id
        self.global_paragraph = global_paragraph
        self.question_text_list = question_text_list
        self.future_doc_tokens_list = future_doc_tokens_list
        self.doc_tokens_list = doc_tokens_list
        self.orig_answer_text_list = orig_answer_text_list
        self.before_state_answer_list = before_state_answer_list
        self.start_position_list = start_position_list
        self.before_state_start_position_list = before_state_start_position_list
        self.end_position_list = end_position_list
        self.before_state_end_position_list = before_state_end_position_list
        self.answer_mask_list = answer_mask_list
        self.before_state_answer_mask_list = before_state_answer_mask_list
        self.known_switch_list = known_switch_list
        self.before_state_known_switch_list = before_state_known_switch_list
        self.action_list = action_list

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text_list)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens_list))
        if self.start_position_list:
            s += ", start_position_list: %d" % (self.start_position_list)
        if self.start_position_list:
            s += ", end_position: %d" % (self.end_position_list)
        return s



class InputFeatures(object):
    """A single set of features of data."""
    global max_num_of_events

    def __init__(self,
                 unique_id_list,
                 example_index_list,
                 doc_span_index_list,
                 tokens_list,
                 token_to_orig_map_list,
                 token_is_max_context_list,
                 input_ids_list,
                 input_mask_list,
                 segment_ids_list,
                 start_position_list=None,
                 before_state_start_position_list=None,
                 end_position_list=None,
                 before_state_end_position_list=None,
                 answer_mask_list=None,
                 before_state_answer_mask_list=None,
                 known_switch_list=None, 
                 before_state_known_switch_list=None, 
                 action_list=None, 
                 unk_mask_list=None,
                 none_mask_list=None,
                 ans_mask_list=None):

        self.unique_id_list = unique_id_list
        self.example_index_list = example_index_list
        self.doc_span_index_list = doc_span_index_list
        self.tokens_list = tokens_list
        self.token_to_orig_map_list = token_to_orig_map_list
        self.token_is_max_context_list = token_is_max_context_list
        self.input_ids_list = input_ids_list
        self.input_mask_list = input_mask_list
        self.segment_ids_list = segment_ids_list
        self.start_position_list = start_position_list
        self.before_state_start_position_list = before_state_start_position_list
        self.end_position_list = end_position_list
        self.before_state_end_position_list = before_state_end_position_list
        self.answer_mask_list = answer_mask_list
        self.before_state_answer_mask_list = before_state_answer_mask_list
        self.known_switch_list = known_switch_list
        self.before_state_known_switch_list = before_state_known_switch_list
        self.action_list=action_list
        self.unk_mask_list = unk_mask_list
        self.none_mask_list = none_mask_list
        self.ans_mask_list = ans_mask_list


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, 'train/high')
        middle = os.path.join(data_dir, 'train/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, 'dev/high')
        middle = os.path.join(data_dir, 'dev/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, 'test/high')
        middle = os.path.join(data_dir, 'test/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw['answers'][i]) - ord('A'))
                question = data_raw['questions'][i]
                options = data_raw['options'][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article], # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
        return examples

class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != 'label':
            raise ValueError(
                "For training, the input file must contain a label column."
            )

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts = [line[4], line[4], line[4], line[4]],
                endings = [line[7], line[8], line[9], line[10]],
                label=line[11]
            ) for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        #There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id = id,
                        question=question,
                        contexts=[options[0]["para"].replace("_", ""), options[1]["para"].replace("_", ""),
                                  options[2]["para"].replace("_", ""), options[3]["para"].replace("_", "")],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth))

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


def find_action_index(before_loc, after_loc): #0=NONE 1=MOVE 2=CREATE 3=DESTROY
  if before_loc == 'none':
    if after_loc == 'none':
      return 0
    else:
      return 2
  elif before_loc == 'unk':
    if after_loc == 'unk':
      return 0
    elif after_loc == "none":
      return 3
    else:
      return 1
  else:
    if after_loc == "none":
      return 3
    elif after_loc == "unk":
      return 1
    elif after_loc != before_loc:
      return 1
    else:
      return 0

def read_propara_examples(input_file, is_training):
    """Read a Propara json file into a list of ProparaExamples."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        global_paragraph = ""
        qas_id = entry["id"]
        locational_prediction_list = entry["local_prediction_data"]

        question_text_list = []
        doc_tokens_list = []
        future_doc_tokens_list = []
        orig_answer_text_list = []
        before_state_answer_list = []
        start_position_list = []
        before_state_start_position_list = []
        end_position_list = []
        before_state_end_position_list = []
        answer_mask_list = []
        before_state_answer_mask_list = []
        known_switch_list = []
        before_state_known_switch_list = []
        action_list = []
        qas_id = entry["id"]
        for locational_entry in locational_prediction_list:
            paragraph_text = locational_entry["context"]
            paragraph_after = locational_entry['context_after']
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            future_doc_tokens = []
            future_char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_after:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        future_doc_tokens.append(c)
                    else:
                        future_doc_tokens[-1] += c
                    prev_is_whitespace = False 
                future_char_to_word_offset.append(len(future_doc_tokens)-1)   


            if len(locational_entry["answer"]) == 0:
                continue            
            question_text = locational_entry["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            answer_mask = None
            known_switch = None
            before_state_start_position = None
            before_state_end_position = None
            before_state_answer_mask = None
            before_state_known_switch = None
            before_state_answer = None
            action = None
            if is_training:

                answer = locational_entry["answer"]
                before_state_answer = locational_entry["answer_before"]

                if before_state_answer == 'unk':
                    before_state_answer_mask = 0
                    before_state_known_switch = 1

                elif before_state_answer == 'none':
                    before_state_answer_mask = 0
                    before_state_known_switch = 2
                else:
                    before_state_answer_mask = 1
                    before_state_known_switch = 0


                orig_answer_text = answer
                if orig_answer_text == 'unk':
                    answer_mask = 0
                    known_switch = 1

                elif orig_answer_text == 'none':
                    answer_mask = 0
                    known_switch = 2
                else:
                    answer_mask = 1
                    known_switch = 0

              
                action = find_action_index(before_state_answer, orig_answer_text)
                answer_offset = int(locational_entry["answer_start"])
                before_state_answer_offset = int(locational_entry["answer_before_start"])

                answer_length = len(orig_answer_text)
                before_state_answer_length = len(before_state_answer)

                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]

                before_state_start_position = char_to_word_offset[before_state_answer_offset]
                before_state_end_position = char_to_word_offset[before_state_answer_offset + before_state_answer_length - 1]

                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))

            question_text_list.append(question_text)
            doc_tokens_list.append(doc_tokens)
            future_doc_tokens_list.append(future_doc_tokens)
            orig_answer_text_list.append(orig_answer_text)
            before_state_answer_list.append(before_state_answer)
            start_position_list.append(start_position)
            before_state_start_position_list.append(before_state_start_position)
            end_position_list.append(end_position)
            before_state_end_position_list.append(before_state_end_position)
            answer_mask_list.append(answer_mask)
            before_state_answer_mask_list.append(before_state_answer_mask)
            known_switch_list.append(known_switch)
            before_state_known_switch_list.append(before_state_known_switch)
            action_list.append(action)


        while(len(question_text_list) < max_num_of_events):
            question_text_list.append("")
            doc_tokens_list.append([])
            orig_answer_text_list.append("")
            before_state_answer_list.append("")
            start_position_list.append(0)
            before_state_start_position_list.append(0)
            end_position_list.append(0)
            before_state_end_position_list.append(0)
            answer_mask_list.append(0)
            before_state_answer_mask_list.append(0)
            known_switch_list.append(0)
            before_state_known_switch_list.append(0)
            action_list.append(0)
            
        example = ProparaExample(
            qas_id=qas_id,
            global_paragraph=global_paragraph,
            question_text_list=question_text_list,
            doc_tokens_list=doc_tokens_list,
            future_doc_tokens_list=future_doc_tokens_list,
            orig_answer_text_list=orig_answer_text_list,
            before_state_answer_list=before_state_answer_list,
            start_position_list=start_position_list,
            before_state_start_position_list=before_state_start_position_list,
            end_position_list=end_position_list,
            before_state_end_position_list=before_state_end_position_list,
            answer_mask_list=answer_mask_list,
            before_state_answer_mask_list=before_state_answer_mask_list,
            known_switch_list=known_switch_list,
            before_state_known_switch_list=before_state_known_switch_list,
            action_list=action_list)

        examples.append(example)
    return examples

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index



def convert_examples_to_features(examples, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=0,
                                 mask_padding_with_zero=True, 
                                 do_full_context=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    #TODO remove labbels
    # label_map = {label : i for i, label in enumerate(label_list)}
    cls_token = tokenizer.cls_token if hasattr(tokenizer, "cls_token") else '[CLS]'
    sep_token = tokenizer.sep_token if hasattr(tokenizer, "sep_token") else '[SEP]'
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0
    
    unique_id = 1000000000
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # choices_features = []

        #My code the features that we need for propara
        tokens_list = []
        token_to_orig_map_list = []
        unk_mask_list = []
        none_mask_list = []
        ans_mask_list = []
        token_is_max_context_list = []
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        start_position_list = []
        before_state_start_position_list = []
        end_position_list = []
        before_state_end_position_list = []
        unique_id_list = []
        example_index_list = []
        doc_span_index_list = []
        paragraph_tok_to_orig_index = []
        paragraph_orig_to_tok_index = []
        paragraph_all_doc_tokens = []
        # for using the full context 
        future_tokens = []
        future_token_to_orig_map = {}
        future_token_is_max_context = {}
        future_segment_ids = []
        future_tokens.append("[CLS]")
        future_segment_ids.append(0)


        for (i, token) in enumerate(example.global_paragraph):
            paragraph_orig_to_tok_index.append(len(paragraph_all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                paragraph_tok_to_orig_index.append(i)
                paragraph_all_doc_tokens.append(sub_token)

        for loc_i in range(len(example.question_text_list)):

            if example.orig_answer_text_list[loc_i] == "":
                unique_id_list.append(unique_id)
                example_index_list.append(ex_index)
                doc_span_index_list.append(0)
                tokens_list.append([])
                token_to_orig_map_list.append({})
                unk_mask_list.append([0 for x in range(max_seq_length)])
                none_mask_list.append([0 for x in range(max_seq_length)])
                ans_mask_list.append([1 for x in range(max_seq_length)])
                token_is_max_context_list.append([])
                input_ids_list.append([0 for x in range(max_seq_length)])
                
                input_mask_list.append([0 for x in range(max_seq_length)])
                segment_ids_list.append([0 for x in range(max_seq_length)])
                start_position_list.append(0)
                before_state_start_position_list.append(0)
                end_position_list.append(0)
                before_state_end_position_list.append(0)
                unique_id += 1
                continue

            query_tokens = tokenizer.tokenize(example.question_text_list[loc_i])
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens_list[loc_i]):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            # added for do_full_context
            future_tok_to_orig_index = []
            future_orig_to_tok_index = []
            future_all_doc_tokens = []
            for (i, token) in enumerate(example.future_doc_tokens_list[loc_i]):
                future_orig_to_tok_index.append(len(future_all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    future_tok_to_orig_index.append(i)
                    future_all_doc_tokens.append(sub_token)

            future_tok_to_orig_index = []
            future_orig_to_tok_index = []
            future_all_doc_tokens = []
            for (i, token) in enumerate(example.future_doc_tokens_list[loc_i]):
                future_orig_to_tok_index.append(len(future_all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    future_tok_to_orig_index.append(i)
                    future_all_doc_tokens.append(sub_token)


            tok_start_position = None
            tok_end_position = None
            if is_training:
                if orig_to_tok_index == []:
                    tok_start_position = 0
                    tok_end_position = 0
                else:
                    tok_start_position = orig_to_tok_index[example.start_position_list[loc_i]]
                    if example.end_position_list[loc_i] < len(example.doc_tokens_list[loc_i]) - 1:
                        tok_end_position = orig_to_tok_index[example.end_position_list[loc_i] + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1
                    (tok_start_position, tok_end_position) = _improve_answer_span(
                        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                        example.orig_answer_text_list[loc_i])


            #added for before locations
            before_state_tok_start_position = None
            before_state_tok_end_position = None
            if is_training:
                if orig_to_tok_index == []:
                    before_state_tok_start_position = 0
                    before_state_tok_start_position = 0
                else:
                    before_state_tok_start_position = orig_to_tok_index[example.before_state_start_position_list[loc_i]]
                    if example.before_state_end_position_list[loc_i] < len(example.doc_tokens_list[loc_i]) - 1:
                        before_state_tok_end_position = orig_to_tok_index[example.before_state_end_position_list[loc_i] + 1] - 1
                    else:
                        before_state_tok_end_position = len(all_doc_tokens) - 1
                    (before_state_tok_start_position, before_state_tok_end_position) = _improve_answer_span(
                        all_doc_tokens, before_state_tok_start_position, before_state_tok_end_position, tokenizer,
                        example.before_state_answer_list[loc_i])

            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            future_doc_spans = []
            future_doc_spans.append(_DocSpan(start=0, length=len(future_all_doc_tokens)))

            
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                if doc_span_index > 0:
                    continue
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []

                tokens.append(cls_token)
                segment_ids.append(0)

                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                    
                tokens.append(sep_token)
                segment_ids.append(0)

                unk_mask = [0 for x in range(max_seq_length)]
                none_mask = [0 for x in range(max_seq_length)]
                ans_mask = [0 for x in range(max_seq_length)]
                
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    if tok_to_orig_index[split_token_index] == 0:
                      unk_mask[len(tokens)] = 1
                      ans_mask[len(tokens)] = 0
                    elif tok_to_orig_index[split_token_index] == 1:
                      ans_mask[len(tokens)] = 0
                      none_mask[len(tokens)] = 1
                    else:
                      ans_mask[len(tokens)] = 1

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)

                if do_full_context == True:
                    tokens.append(sep_token)
                    segment_ids.append(1)
                    for i in range(future_doc_spans[0].length):
                        future_split_token_index = future_doc_spans[0].start + i
                        future_is_max_context = _check_is_max_context(future_doc_spans, 0,
                                                               future_split_token_index)
                        token_is_max_context[len(future_tokens)] = future_is_max_context
                        tokens.append(future_all_doc_tokens[future_split_token_index])
                        segment_ids.append(1)

                tokens.append(sep_token)
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                #for having cls in between sentences we should have 2 dimentional input_mask
                
                input_mask = [1] * len(input_ids)
                
                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                  input_ids.append(0)
                  input_mask.append(0)
                  segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                #######################################################################

                start_position = None
                end_position = None
                if is_training:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    if (example.start_position_list[loc_i] < doc_start or
                            example.end_position_list[loc_i] < doc_start or
                            example.start_position_list[loc_i] > doc_end or example.end_position_list[loc_i] > doc_end):
                        continue

                    doc_offset = len(query_tokens) + 2


                    #the offset is shiftet because of in between cls s
                    doc_start_offset = 0
                    doc_end_offset = 0 
                    for ii in range(0, tok_start_position):
                      split_token_index = doc_span.start + ii
                      if all_doc_tokens[split_token_index] == '.' and (all_doc_tokens[split_token_index-1] != "##k" and all_doc_tokens[split_token_index] != "none"):
                        doc_start_offset += 1
                    for ii in range(0, tok_end_position):
                      split_token_index = doc_span.start + ii
                      if all_doc_tokens[split_token_index] == '.' and (all_doc_tokens[split_token_index-1] != "##k" and all_doc_tokens[split_token_index] != "none"):
                        doc_end_offset += 1
                    start_position = tok_start_position - doc_start + doc_offset 
                    end_position = tok_end_position - doc_start + doc_offset

                #answer span for before state
                before_state_start_position = 0
                before_state_end_position = 0
                if is_training:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    if (example.before_state_start_position_list[loc_i] < doc_start or
                            example.before_state_end_position_list[loc_i] < doc_start or
                            example.before_state_start_position_list[loc_i] > doc_end or example.before_state_end_position_list[loc_i] > doc_end):
                        continue
                    
                    doc_offset = len(query_tokens) + 2

                    before_state_doc_start_offset = 0
                    before_state_doc_end_offset = 0 
                    for ii in range(0, before_state_tok_start_position):
                      split_token_index = doc_span.start + ii
                      if all_doc_tokens[split_token_index] == '.' and (all_doc_tokens[split_token_index-1] != "##k" and all_doc_tokens[split_token_index] != "none"):
                        before_state_doc_start_offset += 1
                    for ii in range(0, before_state_tok_end_position):
                      split_token_index = doc_span.start + ii
                      if all_doc_tokens[split_token_index] == '.' and (all_doc_tokens[split_token_index-1] != "##k" and all_doc_tokens[split_token_index] != "none"):
                        before_state_doc_end_offset += 1

                    before_state_start_position = before_state_tok_start_position - doc_start + doc_offset 
                    before_state_end_position = before_state_tok_end_position - doc_start + doc_offset 

                if ex_index < 20:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (ex_index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info(
                            "answer: %s" % (answer_text))  

            unique_id_list.append(unique_id)
            example_index_list.append(ex_index)
            doc_span_index_list.append(doc_span_index)
            tokens_list.append(tokens)
            token_to_orig_map_list.append(token_to_orig_map)
            token_is_max_context_list.append(token_is_max_context)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            start_position_list.append(start_position)
            before_state_start_position_list.append(before_state_start_position)
            end_position_list.append(end_position)
            before_state_end_position_list.append(before_state_end_position)
            unk_mask_list.append(unk_mask)
            none_mask_list.append(none_mask)
            ans_mask_list.append(ans_mask)
            unique_id += 1

        sum_of_ids =sum([sum(input_ids_list[i]) for i in range(len(input_ids_list))])
        if sum_of_ids == 0:
            continue
        features.append(
            InputFeatures(
                unique_id_list=unique_id_list,     # 1000000000
                example_index_list=example_index_list,       # 0
                doc_span_index_list=doc_span_index_list,     # 0
                tokens_list=tokens_list,          #['cls_token', 'where', 'is', 'magma', '?', 'sep_token', 'un', '##k', '.', 'none', '.', 'future', '_', 'text', '.', 'magma', 'rises', 'from', 'deep', 'in', 'the', 'earth', '.', '[SEP]']
                token_to_orig_map_list=token_to_orig_map_list, # {6: 0, 7: 0, 8: 0, 9: 1, 10: 1, 11: 2, 12: 2, 13: 2, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 7, 20: 8, 21: 9, 22: 9}
                token_is_max_context_list = token_is_max_context_list,
                input_ids_list=input_ids_list,      # [101, 2073, 2003, 28933, 1029, 102, 4895, 2243, 1012, 3904, 1012, 2925, 1035, 3793, 1012, 28933, 9466, 2013, 2784, 1999, 1996, 3011, 1012, 102, 0, 0, 0, 0, 0,
                input_mask_list=input_mask_list,    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                segment_ids_list=segment_ids_list,  # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                start_position_list=start_position_list,   #18
                before_state_start_position_list=before_state_start_position_list,
                end_position_list=end_position_list,       #21
                before_state_end_position_list=before_state_end_position_list,
                answer_mask_list=example.answer_mask_list, # 1
                before_state_answer_mask_list=example.before_state_answer_mask_list,
                known_switch_list=example.known_switch_list,  # 0
                before_state_known_switch_list=example.before_state_known_switch_list, 
                action_list=example.action_list, 
                unk_mask_list=unk_mask_list, 
                none_mask_list=none_mask_list, 
                ans_mask_list=ans_mask_list))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # However, since we'd better not to remove tokens of options and questions, you can choose to use a bigger
    # length or only pop from context
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            logger.info('Attention! you are removing from token_b (swag task is ok). '
                        'If you are training ARC and RACE (you are poping question + options), '
                        'you need to try to use a bigger max seq length!')
            tokens_b.pop()

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits", "before_state_start_logits", "before_state_end_logits", "before_state_switch_logits", "action_logits"])


########################################################################################

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file, output_prediction_file_before_state,output_prediction_file_actions,
                      output_nbest_file, verbose_logging, tokenizer=None):
    
    n_best_size = 20
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        # for j in range(len(feature.example_index_list)):
            example_index_to_features[feature.example_index_list[0]].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "j_index","start_index", "end_index", "start_logit", "end_logit", "switch", 
        "before_state_start_index", "before_state_end_index", "before_state_start_logit", "before_state_end_logit", "before_state_known_switch", "choosen_action"])

    all_predictions = collections.OrderedDict()
    all_predictions_before_state = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_action_json = collections.OrderedDict()
    
    count = -1
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        for j in range(len(feature.example_index_list)):
            prelim_predictions = []
            for (feature_index, feature) in enumerate(features):
          
                if feature.unique_id_list[j] not in unique_id_to_result:
                    continue
                
                result = unique_id_to_result[feature.unique_id_list[j]]
                start_indexes = _get_best_indexes(result.start_logits, n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, n_best_size)
                switch_index = _get_best_indexes(result.switch_logits, 1)

                before_state_start_indexes = _get_best_indexes(result.before_state_start_logits[6:], n_best_size)
                before_state_end_indexes = _get_best_indexes(result.before_state_end_logits[6:], n_best_size)
                before_state_switch_index = _get_best_indexes(result.before_state_switch_logits, 1)

                choosen_action = _get_best_indexes(result.action_logits, 1)
                best_before_state_start_index = -1
                best_before_state_end_index = -1
                for before_state_start_index in before_state_start_indexes:
                    before_state_start_index = before_state_start_index + 6
                    for before_state_end_index in before_state_end_indexes:
                        before_state_end_index = before_state_end_index + 6
                        if before_state_start_index >= len(feature.tokens_list[j]):
                            continue
                        if before_state_end_index >= len(feature.tokens_list[j]):
                            continue
                        if before_state_start_index not in feature.token_to_orig_map_list[j]:
                            continue
                        if before_state_end_index not in feature.token_to_orig_map_list[j]:
                            continue
                        if not feature.token_is_max_context_list[j].get(before_state_start_index, False):
                            continue
                        if before_state_end_index < before_state_start_index:
                            before_state_end_index = before_state_start_index
                            continue
                        length = before_state_end_index - before_state_start_index + 1
                        if length > 2:
                            continue
                        best_before_state_start_index = before_state_start_index
                        best_before_state_end_index = before_state_end_index
                        break
                    if best_before_state_start_index != -1:
                        break

                if best_before_state_start_index == -1:
                    for before_state_start_index in before_state_start_indexes:
                        before_state_start_index = before_state_start_index + 6
                        for before_state_end_index in before_state_end_indexes:
                            before_state_end_index = before_state_end_index + 6
                            if before_state_start_index >= len(feature.tokens_list[j]):
                                continue
                            if before_state_end_index >= len(feature.tokens_list[j]):
                                continue
                            if before_state_start_index not in feature.token_to_orig_map_list[j]:
                                continue
                            if before_state_end_index not in feature.token_to_orig_map_list[j]:
                                continue
                            if not feature.token_is_max_context_list[j].get(before_state_start_index, False):
                                continue
                            if before_state_end_index < before_state_start_index:
                                before_state_end_index = before_state_start_index
                            length = before_state_end_index - before_state_start_index + 1
                            if length > 2:
                                before_state_end_index = before_state_start_index + 2
                            best_before_state_start_index = before_state_start_index
                            best_before_state_end_index = before_state_end_index
                            break
                        if best_before_state_start_index != -1:
                            break

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens_list[j]):
                            continue
                        if end_index >= len(feature.tokens_list[j]):
                            continue
                        if start_index not in feature.token_to_orig_map_list[j]:
                            continue
                        if end_index not in feature.token_to_orig_map_list[j]:
                            continue
                        if end_index < start_index:
                            end_index = start_index
                            continue
                        length = end_index - start_index + 1
                        if length > 5:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                j_index=j,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index], 
                                switch=switch_index[0], 
                                before_state_start_index=before_state_start_index, 
                                before_state_end_index=before_state_end_index,
                                before_state_start_logit=result.before_state_start_logits[before_state_start_index], 
                                before_state_end_logit=result.before_state_end_logits,
                                before_state_known_switch=before_state_switch_index[0], 
                                choosen_action=choosen_action[0]))

        
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "j_index", "start_logit", "end_logit", "switch", "before_state_text", "before_state_start_logit", "before_state_end_logit", "before_state_known_switch", "choosen_action"])

                seen_predictions = {}
                nbest = []
                for pred in prelim_predictions:
                    if len(nbest) >= n_best_size:
                        break
                    feature = features[pred.feature_index]
                    tok_tokens = feature.tokens_list[pred.j_index][pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map_list[pred.j_index][pred.start_index]
                    orig_doc_end = feature.token_to_orig_map_list[pred.j_index][pred.end_index]
                    orig_tokens = example.doc_tokens_list[pred.j_index][orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, tokenizer=tokenizer)
                    if len(final_text) == 1 or final_text == 'an' or final_text == "the":
                        continue
                    try:

                        before_state_tok_tokens = feature.tokens_list[pred.j_index][int(pred.before_state_start_index):(pred.before_state_end_index + 1)]
                        before_state_orig_doc_start = feature.token_to_orig_map_list[pred.j_index][pred.before_state_start_index]
                        before_state_orig_doc_end = feature.token_to_orig_map_list[pred.j_index][pred.before_state_end_index]
                        before_state_orig_tokens = example.doc_tokens_list[pred.j_index][before_state_orig_doc_start:(before_state_orig_doc_end + 1)]
                        before_state_tok_text = " ".join(before_state_tok_tokens)

                        # De-tokenize WordPieces that have been split off.
                        before_state_tok_text = before_state_tok_text.replace(" ##", "")
                        before_state_tok_text = before_state_tok_text.replace("##", "")

                        # Clean whitespace
                        before_state_tok_text = before_state_tok_text.strip()
                        before_state_tok_text = " ".join(before_state_tok_text.split())
                        before_state_orig_text = " ".join(before_state_orig_tokens)

                        before_state_final_text = get_final_text(before_state_tok_text, before_state_orig_text, do_lower_case, verbose_logging)
                    except:
                        before_state_final_text = "empty"
                    if (final_text, before_state_final_text) in seen_predictions:
                        continue



                    seen_predictions[(final_text, before_state_final_text)] = True
                    nbest.append(
                        _NbestPrediction(
                            text=final_text,
                            j_index=pred.j_index,
                            start_logit=pred.start_logit,
                            end_logit=pred.end_logit,
                            switch=pred.switch, 
                            before_state_text=before_state_final_text,
                            before_state_start_logit=pred.before_state_start_logit, 
                            before_state_end_logit=pred.before_state_end_logit,
                            before_state_known_switch=pred.before_state_known_switch,
                            choosen_action=pred.choosen_action))

                # In very rare edge cases we could have no valid predictions. So we
                # just create a nonce prediction in this case to avoid failure.
                if not nbest:
                    nbest.append(
                        _NbestPrediction(text="empty", j_index=j, start_logit=0.0, end_logit=0.0, switch=0.0, before_state_text="empty", before_state_start_logit=0.0, before_state_end_logit=0.0, before_state_known_switch=0.0, choosen_action=0.0))

                assert len(nbest) >= 1

                total_scores = []
                for entry in nbest:
                    total_scores.append(entry.start_logit + entry.end_logit)

                probs = _compute_softmax(total_scores)

                nbest_json = []
                for (i, entry) in enumerate(nbest):
                    output = collections.OrderedDict()
                    output["text"] = entry.text
                    output["j_index"] = entry.j_index
                    output["probability"] = probs[i]
                    output["start_logit"] = entry.start_logit
                    output["end_logit"] = entry.end_logit
                    output["switch"]= entry.switch
                    output["before_state_text"] = entry.before_state_text
                    output["before_state_start_logit"] = entry.before_state_start_logit
                    output["before_state_end_logit"] = entry.before_state_end_logit
                    output["before_state_known_switch"]= entry.before_state_known_switch
                    output["choosen_action"] = entry.choosen_action
                    nbest_json.append(output)

                assert len(nbest_json) >= 1
                if nbest_json[0]['switch'] == 1:
                    all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = "none"
                    nbest_json[0]['text'] = "none"
                elif nbest_json[0]['switch'] == 2:
                    all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = "unk"
                    nbest_json[0]['text'] = "unk"
                else:
                    all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = nbest_json[0]["text"]
                    if all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])].startswith("none") or all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])].startswith("unk"):
                        # print(all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])])
                        if all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] == "unk." or all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] == "unk":
                            all_predictions[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = "none"

                if nbest_json[0]['before_state_known_switch'] == 1:
                    all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = "none"
                    nbest_json[0]['before_state_text'] = "none"
                elif nbest_json[0]['before_state_known_switch'] == 2:
                    all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = "unk"
                    nbest_json[0]['before_state_text'] = "unk"
                else:
                    all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] =  nbest_json[0]["before_state_text"]
                    if all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])].startswith("none") or all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])].startswith("unk"):
                        print("here")
                        # if  all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])]== "none. the piston":
                        #      all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])]="the piston"
                        print(all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])])
                        # if all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] == "none." or all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] == "none":
                        #     all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = "unk"
                        if all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] == "unk." or all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] == "unk":
                            all_predictions_before_state[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = "none"


                # all_nbest_json[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = nbest_json
                all_action_json[str(example.qas_id) + '+' + str(nbest_json[0]["j_index"])] = nbest_json[0]["choosen_action"]

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_prediction_file_actions, "w") as writer:
        writer.write(json.dumps(all_action_json, indent=4) + "\n")

    with open(output_prediction_file_before_state, "w") as writer:
        writer.write(json.dumps(all_predictions_before_state, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")



def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False, tokenizer=None):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    if tokenizer == None:
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs



processors = {
    "race": RaceProcessor,
    "swag": SwagProcessor,
    "arc": ArcProcessor
}


GLUE_TASKS_NUM_LABELS = {
    "race", 4,
    "swag", 4,
    "arc", 4
}
