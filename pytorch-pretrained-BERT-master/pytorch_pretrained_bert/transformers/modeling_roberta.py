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
"""PyTorch RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable

from pytorch_pretrained_bert.transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from pytorch_pretrained_bert.transformers.configuration_roberta import RobertaConfig
# from pytorch_pretrained_bert.transformers.file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids)


ROBERTA_START_DOCSTRING = r"""    The RoBERTa model was proposed in
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
    by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
    Veselin Stoyanov. It is based on Google's BERT model released in 2018.
    
    It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
    objective and training with much larger mini-batches and learning rates.
    
    This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained 
    models.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the 
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> Is this Jacksonville ? </s> </s> No it is not . </s>``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

            Fully encoded sequences or sequence pairs can be obtained using the RobertaTokenizer.encode function with 
            the ``add_special_tokens`` parameter set to ``True``.

            RoBERTa is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional` need to be trained) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional segment token indices to indicate first and second portions of the inputs.
            This embedding matrice is not trained (not pretrained during RoBERTa pretraining), you will have to train it
            during finetuning.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1[``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

# @add_start_docstrings("The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
#                       ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if input_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")
        return super(RobertaModel, self).forward(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 position_ids=position_ids,
                                                 head_mask=head_mask)


# @add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """,
#     ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head.decoder, self.roberta.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x


# @add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer 
#     on top of the pooled output) e.g. for GLUE tasks. """,
#     ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

# @add_start_docstrings("""Roberta Model with a multiple choice classification head on top (a linear layer on top of
#     the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
#     ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForMultipleChoice(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# @add_start_docstrings("""Roberta Model with a multiple choice classification head on top (a linear layer on top of
#     the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
#     ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
# class RobertaForMultipleChoice(BertPreTrainedModel):

#     config_class = RobertaConfig
#     pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
#     base_model_prefix = "roberta"

#     def __init__(self, config):
#         super(RobertaForMultipleChoice, self).__init__(config)

#         self.roberta = RobertaModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)

#         self.init_weights()

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
#                 position_ids=None, head_mask=None):
#         num_choices = input_ids.shape[1]

#         flat_input_ids = input_ids.view(-1, input_ids.size(-1))
#         flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
#         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
#                             attention_mask=flat_attention_mask, head_mask=head_mask)
#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)

#         outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             outputs = (loss,) + outputs

#         return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

class action_decoder(nn.Module):
    def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
        super(action_decoder, self).__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_class = n_class




class Roberta_GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one(BertPreTrainedModel):
    
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    config_class = RobertaConfig

    def __init__(self, config, n_class=3, hidden_dim=1000, sequence_length=200):
        super(Roberta_GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one, self).__init__(config)
        self.batch_size = 8
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
        self.roberta = RobertaModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        
        self.lstm_for_tagging = nn.LSTM(config.hidden_size, self.hidden_dim, bidirectional=True,  batch_first=True)
        self.lstm_for_start= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
        self.lstm_for_start_before= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
    
        self.hidden2tag = nn.Linear(hidden_dim*sequence_length*2, n_class)
        self.hidden2tag_before_state = nn.Linear(hidden_dim*sequence_length*2, n_class)
    
        ########################################

        self.start_score_lstm = nn.Linear(hidden_dim*2, 1).cuda()
        self.end_score_lstm = nn.Linear(hidden_dim*2, 1).cuda()
        self.start_score_lstm_before = nn.Linear(hidden_dim*2, 1).cuda()
        self.end_score_lstm_before = nn.Linear(hidden_dim*2, 1).cuda()
        
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax()
        self.ReLU = nn.ReLU()

        self.hidden_end = self.init_hidden(self.batch_size)
        self.hidden_start = self.init_hidden(self.batch_size)
        self.hidden_start_before = self.init_hidden(self.batch_size)
        self.hidden_end_before = self.init_hidden(self.batch_size)
        self.hidden = self.init_hidden(self.batch_size)
        
        self.init_weights()
        

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.hidden_start = self.init_hidden(self.batch_size)
        self.hidden_end = self.init_hidden(self.batch_size)
        self.hidden_start_before = self.init_hidden(self.batch_size)
        self.hidden_end_before = self.init_hidden(self.batch_size)
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batchsize):
        return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
                torch.zeros(2, batchsize, self.hidden_dim).cuda())

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, 
                                before_state_start_positions=None, before_state_end_positions=None, before_state_answer_mask=None,before_state_known_switch=None, weight_class_loss=1, weight_span_loss =1, actions=None):
    


        if start_positions is not None and end_positions is not None: # training mode

            start_logits_list = []
            end_logits_list = []
            switch_logits_list = []
            before_state_start_logits_list = []
            before_state_end_logits_list = []
            before_state_switch_logits_list = []
            sequence_list = []

            self.batch_size = len(input_ids)
            self.hidden = self.init_hidden(self.batch_size)
            self.hidden_start_before = self.init_hidden(self.batch_size)
            self.hidden_start = self.init_hidden(self.batch_size)

            self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
            self.hidden_start_before = (Variable(self.hidden_start_before[0].data, requires_grad=True), Variable(self.hidden_start_before[1].data, requires_grad=True))
            self.hidden_start = (Variable(self.hidden_start[0].data, requires_grad=True), Variable(self.hidden_start[1].data, requires_grad=True))
            
            final_loss = torch.zeros(1).cuda()
            padding_flag = [1 for x in range(self.batch_size)]
            
            for i in range(len(input_ids[0])):
                for j in range(len(input_ids)):
                    if sum(input_ids[j][i]).cpu().numpy() == 0:
                        padding_flag[j] = 0 
                
                # import pdb; pdb.set_trace()
                outputs = self.roberta(input_ids[:,i].clone(), attention_mask=attention_mask[:,i].clone(),token_type_ids=token_type_ids[:,i].clone())
                # import pdb; pdb.set_trace()
                sequence_output = outputs[0]
                sequence_list.append(sequence_output)
                
                lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
                lstm_out = self.ReLU(lstm_out)
                start_logits = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
                end_logits = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
                
                lstm_out_b, self.hidden_start_before = self.lstm_for_start_before(sequence_output, self.hidden_start_before)
                lstm_out_b = self.ReLU(lstm_out_b)
                before_state_start_logits = self.start_score_lstm_before(lstm_out_b.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
                before_state_end_logits = self.end_score_lstm_before(lstm_out_b.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)

                lstm_out_switch, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
                lstm_out_switch = self.ReLU(lstm_out_switch)
                switch_logits = self.hidden2tag(lstm_out_switch.contiguous().view(-1, 2*self.sequence_length*self.hidden_dim))
                before_state_switch_logits = self.hidden2tag_before_state(lstm_out_switch.contiguous().view(-1, 2*self.sequence_length*self.hidden_dim))


                if len(start_positions[:,i].clone().size()) > 1:
                    start_positions[:,i] = start_positions[:,i].clone().squeeze(-1)
                if len(end_positions[:,i].clone().size()) > 1:
                    end_positions[:,i] = end_positions[:,i].clone().squeeze(-1)
                answer_mask[:,i] = answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
                span_mask = (known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()
                before_state_answer_mask[:,i] = before_state_answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
                before_state_span_mask = (before_state_known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()

                ignored_index = start_logits.size(1)
                start_positions[:,i].clone().clamp_(0, ignored_index)
                end_positions[:,i].clone().clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(reduce=False)
                start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions[:,i].clone(), span_mask)]
                start_losses = [(loss_fct(start_logits[ii].unsqueeze(0), start_pairs[ii][0].unsqueeze(0)) * start_pairs[ii][1] * padding_flag[ii]) for ii in range(len(start_pairs))]
                
                end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions[:,i].clone(), span_mask)]
                end_losses = [(loss_fct(end_logits[ii].unsqueeze(0), end_pairs[ii][0].unsqueeze(0)) * end_pairs[ii][1]* padding_flag[ii]) for ii in range(len(end_pairs))]
                
                switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch[:,i].clone(), answer_mask[:,i].clone())]
                switch_losses = [(loss_fct(switch_logits[ii].unsqueeze(0), switch_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(switch_pairs))]


                before_state_start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(before_state_start_positions[:,i].clone(), before_state_span_mask)]
                before_state_start_losses = [(loss_fct(before_state_start_logits[ii].unsqueeze(0), before_state_start_pairs[ii][0].unsqueeze(0)) * before_state_start_pairs[ii][1] * padding_flag[ii]) for ii in range(len(before_state_start_pairs))]
                
                before_state_end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(before_state_end_positions[:,i].clone(), before_state_span_mask)]
                before_state_end_losses = [(loss_fct(before_state_end_logits[ii].unsqueeze(0), before_state_end_pairs[ii][0].unsqueeze(0)) * before_state_end_pairs[ii][1]* padding_flag[ii]) for ii in range(len(before_state_end_pairs))]
                
                before_state_switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(before_state_known_switch[:,i].clone(), before_state_answer_mask[:,i].clone())]
                before_state_switch_losses = [(loss_fct(before_state_switch_logits[ii].unsqueeze(0), before_state_switch_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(before_state_switch_pairs))]
                total_loss = torch.sum(sum((sum(start_losses + end_losses + before_state_start_losses + before_state_end_losses) * weight_span_loss) + (sum(switch_losses + before_state_switch_losses) * weight_class_loss)))
                


                start_probs = self.softmax(start_logits)           
                end_probs = self.softmax(end_logits)
                switch_probs = self.softmax(switch_logits)
                before_state_start_probs = self.softmax(before_state_start_logits)
                before_state_end_probs = self.softmax(before_state_end_logits)
                before_state_switch_probs = self.softmax(before_state_switch_logits)


                start_logits_list.append(start_probs)
                end_logits_list.append(end_probs)
                switch_logits_list.append(switch_probs)
                before_state_start_logits_list.append(before_state_start_probs)
                before_state_end_logits_list.append(before_state_end_probs)
                before_state_switch_logits_list.append(before_state_switch_probs)

                final_loss += total_loss
            return final_loss, start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list, []

        else:
            self.batch_size = len(input_ids)
            self.hidden = self.init_hidden(self.batch_size)
            self.hidden_start_before = self.init_hidden(self.batch_size)
            self.hidden_start = self.init_hidden(self.batch_size)
            
            sequence_list = []
            start_logits_list = []
            end_logits_list = []
            switch_logits_list = []
            before_state_start_logits_list = []
            before_state_end_logits_list = []
            before_state_switch_logits_list = []
            
            for i in range(len(input_ids[0])):

                outputs = self.roberta(input_ids[:,i].clone(), attention_mask=attention_mask[:,i].clone(),token_type_ids=token_type_ids[:,i].clone())
                sequence_output = outputs[0]
                sequence_list.append(sequence_output)
                
                lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
                lstm_out = self.ReLU(lstm_out)
                start_logits = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
                end_logits = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
                
                lstm_out_b, self.hidden_start_before = self.lstm_for_start_before(sequence_output, self.hidden_start_before)
                lstm_out_b = self.ReLU(lstm_out_b)
                before_state_start_logits = self.start_score_lstm_before(lstm_out_b.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
                before_state_end_logits = self.end_score_lstm_before(lstm_out_b.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)

                lstm_out_switch, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
                lstm_out_switch = self.ReLU(lstm_out_switch)
                switch_logits = self.hidden2tag(lstm_out_switch.contiguous().view(-1, 2*self.sequence_length*self.hidden_dim))
                before_state_switch_logits = self.hidden2tag_before_state(lstm_out_switch.contiguous().view(-1, 2*self.sequence_length*self.hidden_dim))


                start_probs = self.softmax(start_logits)           
                end_probs = self.softmax(end_logits)
                switch_probs = self.softmax(switch_logits)
                before_state_start_probs = self.softmax(before_state_start_logits)
                before_state_end_probs = self.softmax(before_state_end_logits)
                before_state_switch_probs = self.softmax(before_state_switch_logits)


                start_logits_list.append(start_probs)
                end_logits_list.append(end_probs)
                switch_logits_list.append(switch_probs)
                before_state_start_logits_list.append(before_state_start_probs)
                before_state_end_logits_list.append(before_state_end_probs)
                before_state_switch_logits_list.append(before_state_switch_probs)


            return start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list, []




class Roberta_Sequence_decoder(action_decoder):

    def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
        super(Roberta_Sequence_decoder, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)

        self.rnn = nn.LSTM(self.input_dim * 5, self.hidden_dim, bidirectional=True, batch_first=True)
        self.non_linear = nn.ReLU()
        self.action_classifier = nn.Linear(self.hidden_dim*2 * self.sequence_length, self.n_class)
        self.hidden = self.init_hidden(self.batch_size)
        
    def init_hidden(self, batchsize):
        return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
                torch.zeros(2, batchsize, self.hidden_dim).cuda())


    def forward(self, input_ids, sequence_list , start_probs=None, end_probs=None, classification_probs=None, 
                            before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, 
                            actions=None, unk_mask=None, none_mask=None, ans_mask=None):
        
        if actions is not None:  #training
            self.batch_size = len(input_ids)  # the last batch might be incomplete
            self.hidden = self.init_hidden(self.batch_size)
            self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
            
            final_loss = torch.zeros(1).cuda()
            before_state_start = None
            before_state_end =None
            padding_flag = [1 for x in range(self.batch_size)]     # padding that is used for masking out loss of padded inputs. 
            
            for i in range(len(input_ids[0])):
                for j in range(len(input_ids)):
                    if sum(input_ids[j][i]).cpu().numpy() == 0:
                        padding_flag[j] = 0

                weigthed_start_ans = torch.mul((start_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_unk = torch.mul((start_probs[i] * unk_mask[:,i].clone()), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_none = torch.mul((start_probs[i] * none_mask[:,i].clone()), classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                start_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])

                weigthed_start_ans = torch.mul((end_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_unk = torch.mul((end_probs[i] *unk_mask[:,i]).clone(), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_none = torch.mul((end_probs[i] *none_mask[:,i]).clone(), classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                end_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])

                if i == 0:
                    weigthed_start_ans = torch.mul((before_state_start_probs[i] * ans_mask[:,i].clone()) ,before_state_classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_unk = torch.mul((before_state_start_probs[i] * unk_mask[:,i].clone()), before_state_classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_none = torch.mul((before_state_start_probs[i] * none_mask[:,i].clone()), before_state_classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    before_state_start_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])


                    weigthed_start_ans = torch.mul((before_state_end_probs[i] * ans_mask[:,i].clone()), before_state_classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_unk = torch.mul((before_state_end_probs[i] * unk_mask[:,i].clone()), before_state_classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_none = torch.mul((before_state_end_probs[i] * none_mask[:,i].clone()), before_state_classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    before_state_end_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])

                    rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start_probs_with_attention, before_state_end_probs_with_attention), 2)
                
                else:
                    rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start, before_state_end), 2)

                before_state_end = end_probs_with_attention
                before_state_start = start_probs_with_attention
                
                rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
                action_logits = self.non_linear(rnn_output)
                action_logits = self.action_classifier(action_logits.contiguous().view(-1, self.sequence_length*2* self.hidden_dim))
                
                loss_fct = CrossEntropyLoss(reduce=False)
                action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
                action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

                final_loss += torch.sum(sum(action_losses))
            return final_loss
        
        else:   #evaluaiton phase

            self.batch_size = len(input_ids)
            self.hidden = self.init_hidden(self.batch_size)
            self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

            action_logits_list = []
            before_state_start = None
            before_state_end = None
            
            for i in range(len(input_ids[0])):
                weigthed_start_ans = torch.mul((start_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_unk = torch.mul((start_probs[i] * unk_mask[:,i].clone()), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_none = torch.mul((start_probs[i] * none_mask[:,i].clone()), classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                start_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])

                weigthed_start_ans = torch.mul((end_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_unk = torch.mul((end_probs[i] *unk_mask[:,i]).clone(), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                weigthed_start_none = torch.mul((end_probs[i] *none_mask[:,i]).clone(), classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                end_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])

                if i == 0:
                    weigthed_start_ans = torch.mul((before_state_start_probs[i] * ans_mask[:,i].clone()) ,before_state_classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_unk = torch.mul((before_state_start_probs[i] * unk_mask[:,i].clone()), before_state_classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_none = torch.mul((before_state_start_probs[i] * none_mask[:,i].clone()), before_state_classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    before_state_start_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])


                    weigthed_start_ans = torch.mul((before_state_end_probs[i] * ans_mask[:,i].clone()), before_state_classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_unk = torch.mul((before_state_end_probs[i] * unk_mask[:,i].clone()), before_state_classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    weigthed_start_none = torch.mul((before_state_end_probs[i] * none_mask[:,i].clone()), before_state_classification_probs[i][:,2].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
                    before_state_end_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) + torch.mul(weigthed_start_none.unsqueeze(2), sequence_list[i])

                    rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start_probs_with_attention, before_state_end_probs_with_attention), 2)
                
                else:
                    rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start, before_state_end), 2)

                before_state_end = end_probs_with_attention
                before_state_start = start_probs_with_attention

                rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
                action_logits = self.non_linear(rnn_output)
                action_logits = self.action_classifier(action_logits.contiguous().view(-1, self.sequence_length*2* self.hidden_dim))

                action_logits_list.append(action_logits)
            return action_logits_list

class Roberta_span_class_action_prediction(BertPreTrainedModel):

    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    config_class = RobertaConfig

    def __init__(self, config, n_class=3, hidden_dim=1000, sequence_length=200, n_actions=4):
        super(Roberta_span_class_action_prediction, self).__init__(config)
        self.batch_size = 8
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.bert_hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version

        self.hidden2tag = nn.Linear(config.hidden_size*sequence_length, n_class)
        self.hidden2tag_before_state = nn.Linear(config.hidden_size*sequence_length, n_class)
        self.action_classifier = nn.Linear(config.hidden_size * self.sequence_length, n_actions)

        ########################################

        self.start_score = nn.Linear(config.hidden_size, 1).cuda()
        self.end_score = nn.Linear(config.hidden_size, 1).cuda()
        self.start_score_before = nn.Linear(config.hidden_size, 1).cuda()
        self.end_score_before = nn.Linear(config.hidden_size, 1).cuda()
        self.softmax = nn.Softmax()
        self.init_weights()
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, 
                                before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None, weight_class_loss=1, weight_span_loss =1, weight_action_loos=1, actions=None):

        if start_positions is not None and end_positions is not None: # training mode

            start_logits_list = []
            end_logits_list = []
            switch_logits_list = []
            before_state_start_logits_list = []
            before_state_end_logits_list = []
            before_state_switch_logits_list = []
            action_logits_list = []
            sequence_list = []

            self.batch_size = len(input_ids)
            
            final_loss = torch.zeros(1).cuda()
            padding_flag = [1 for x in range(self.batch_size)]
            
            for i in range(len(input_ids[0])):
                for j in range(len(input_ids)):
                    if sum(input_ids[j][i]).cpu().numpy() == 0:
                        padding_flag[j] = 0 
                

                outputs = self.roberta(input_ids[:,i].clone(), attention_mask=attention_mask[:,i].clone(),token_type_ids=token_type_ids[:,i].clone())
                sequence_output = outputs[0]

                sequence_list.append(sequence_output)
                
                start_logits = self.start_score(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)
                end_logits = self.end_score(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)
                
                before_state_start_logits = self.start_score_before(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)
                before_state_end_logits = self.end_score_before(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)

                switch_logits = self.hidden2tag(sequence_output.contiguous().view(-1, self.sequence_length*self.bert_hidden_size))
                before_state_switch_logits = self.hidden2tag_before_state(sequence_output.contiguous().view(-1, self.sequence_length*self.bert_hidden_size))

                action_logits = self.action_classifier(sequence_output.contiguous().view(-1, self.sequence_length*self.bert_hidden_size))
                

                if len(start_positions[:,i].clone().size()) > 1:
                    start_positions[:,i] = start_positions[:,i].clone().squeeze(-1)
                if len(end_positions[:,i].clone().size()) > 1:
                    end_positions[:,i] = end_positions[:,i].clone().squeeze(-1)
                
                answer_mask[:,i] = answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
                span_mask = (known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()
                before_state_answer_mask[:,i] = before_state_answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
                before_state_span_mask = (before_state_known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()

                ignored_index = start_logits.size(1)
                start_positions[:,i].clone().clamp_(0, ignored_index)
                end_positions[:,i].clone().clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(reduce=False)
                start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions[:,i].clone(), span_mask)]
                start_losses = [(loss_fct(start_logits[ii].unsqueeze(0), start_pairs[ii][0].unsqueeze(0)) * start_pairs[ii][1] * padding_flag[ii]) for ii in range(len(start_pairs))]
                
                end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions[:,i].clone(), span_mask)]
                end_losses = [(loss_fct(end_logits[ii].unsqueeze(0), end_pairs[ii][0].unsqueeze(0)) * end_pairs[ii][1]* padding_flag[ii]) for ii in range(len(end_pairs))]
                
                switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch[:,i].clone(), answer_mask[:,i].clone())]
                switch_losses = [(loss_fct(switch_logits[ii].unsqueeze(0), switch_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(switch_pairs))]


                before_state_start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(before_state_start_positions[:,i].clone(), before_state_span_mask)]
                before_state_start_losses = [(loss_fct(before_state_start_logits[ii].unsqueeze(0), before_state_start_pairs[ii][0].unsqueeze(0)) * before_state_start_pairs[ii][1] * padding_flag[ii]) for ii in range(len(before_state_start_pairs))]
                
                before_state_end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(before_state_end_positions[:,i].clone(), before_state_span_mask)]
                before_state_end_losses = [(loss_fct(before_state_end_logits[ii].unsqueeze(0), before_state_end_pairs[ii][0].unsqueeze(0)) * before_state_end_pairs[ii][1]* padding_flag[ii]) for ii in range(len(before_state_end_pairs))]
                
                before_state_switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(before_state_known_switch[:,i].clone(), before_state_answer_mask[:,i].clone())]
                before_state_switch_losses = [(loss_fct(before_state_switch_logits[ii].unsqueeze(0), before_state_switch_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(before_state_switch_pairs))]
                
                action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
                action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

                span_loss = torch.sum(sum(start_losses + end_losses + before_state_start_losses + before_state_end_losses) * weight_span_loss)
                class_loss = torch.sum(sum(switch_losses + before_state_switch_losses) * weight_class_loss)
                action_loss = torch.sum(sum(action_losses) * weight_action_loos)
                total_loss = torch.sum(span_loss + action_loss + class_loss)
                


                start_probs = self.softmax(start_logits)           
                end_probs = self.softmax(end_logits)
                switch_probs = self.softmax(switch_logits)
                before_state_start_probs = self.softmax(before_state_start_logits)
                before_state_end_probs = self.softmax(before_state_end_logits)
                before_state_switch_probs = self.softmax(before_state_switch_logits)
                action_probs = self.softmax(action_logits)

                start_logits_list.append(start_probs)
                end_logits_list.append(end_probs)
                switch_logits_list.append(switch_probs)
                before_state_start_logits_list.append(before_state_start_probs)
                before_state_end_logits_list.append(before_state_end_probs)
                before_state_switch_logits_list.append(before_state_switch_probs)
                action_logits_list.append(action_probs)

                final_loss += total_loss
            return final_loss, start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list, action_logits_list

        else:
            self.batch_size = len(input_ids)
            sequence_list = []
            start_logits_list = []
            end_logits_list = []
            switch_logits_list = []
            before_state_start_logits_list = []
            before_state_end_logits_list = []
            before_state_switch_logits_list = []
            action_logits_list = []
            
            for i in range(len(input_ids[0])):
                outputs = self.roberta(input_ids[:,i].clone(), attention_mask=attention_mask[:,i].clone(),token_type_ids=token_type_ids[:,i].clone())
                sequence_output = outputs[0]

                sequence_list.append(sequence_output)
                
                start_logits = self.start_score(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)
                end_logits = self.end_score(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)
                
                before_state_start_logits = self.start_score_before(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)
                before_state_end_logits = self.end_score_before(sequence_output.contiguous().view(-1, self.sequence_length, self.bert_hidden_size)).squeeze(-1)

                switch_logits = self.hidden2tag(sequence_output.contiguous().view(-1, self.sequence_length*self.bert_hidden_size))
                before_state_switch_logits = self.hidden2tag_before_state(sequence_output.contiguous().view(-1, self.sequence_length*self.bert_hidden_size))

                action_logits = self.action_classifier(sequence_output.contiguous().view(-1, self.sequence_length*self.bert_hidden_size))
                

                start_probs = self.softmax(start_logits)           
                end_probs = self.softmax(end_logits)
                switch_probs = self.softmax(switch_logits)
                before_state_start_probs = self.softmax(before_state_start_logits)
                before_state_end_probs = self.softmax(before_state_end_logits)
                before_state_switch_probs = self.softmax(before_state_switch_logits)
                action_probs = self.softmax(action_logits)

                start_logits_list.append(start_probs)
                end_logits_list.append(end_probs)
                switch_logits_list.append(switch_probs)
                before_state_start_logits_list.append(before_state_start_probs)
                before_state_end_logits_list.append(before_state_end_probs)
                before_state_switch_logits_list.append(before_state_switch_probs)
                action_logits_list.append(action_probs)

            return start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list, action_logits_list


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
