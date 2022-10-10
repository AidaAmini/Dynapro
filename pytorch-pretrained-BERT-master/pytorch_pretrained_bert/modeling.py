# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.autograd import Variable

from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
	'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
	'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
	'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
	'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
	'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
	'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
	'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
	"scibert-scivocab-uncased": "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
	return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
	"""Configuration class to store the configuration of a `BertModel`.
	"""
	def __init__(self,
				 vocab_size_or_config_json_file,
				 hidden_size=768,
				 num_hidden_layers=12,
				 num_attention_heads=12,
				 intermediate_size=3072,
				 hidden_act="gelu",
				 hidden_dropout_prob=0.1,
				 attention_probs_dropout_prob=0.1,
				 max_position_embeddings=512,
				 type_vocab_size=2,
				 initializer_range=0.02):
		"""Constructs BertConfig.

		Args:
			vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
			hidden_size: Size of the encoder layers and the pooler layer.
			num_hidden_layers: Number of hidden layers in the Transformer encoder.
			num_attention_heads: Number of attention heads for each attention layer in
				the Transformer encoder.
			intermediate_size: The size of the "intermediate" (i.e., feed-forward)
				layer in the Transformer encoder.
			hidden_act: The non-linear activation function (function or string) in the
				encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
			hidden_dropout_prob: The dropout probabilitiy for all fully connected
				layers in the embeddings, encoder, and pooler.
			attention_probs_dropout_prob: The dropout ratio for the attention
				probabilities.
			max_position_embeddings: The maximum sequence length that this model might
				ever be used with. Typically set this to something large just in case
				(e.g., 512 or 1024 or 2048).
			type_vocab_size: The vocabulary size of the `token_type_ids` passed into
				`BertModel`.
			initializer_range: The sttdev of the truncated_normal_initializer for
				initializing all weight matrices.
		"""
		if isinstance(vocab_size_or_config_json_file, str):
			with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
				json_config = json.loads(reader.read())
			for key, value in json_config.items():
				self.__dict__[key] = value
		elif isinstance(vocab_size_or_config_json_file, int):
			self.vocab_size = vocab_size_or_config_json_file
			self.hidden_size = hidden_size
			self.num_hidden_layers = num_hidden_layers
			self.num_attention_heads = num_attention_heads
			self.hidden_act = hidden_act
			self.intermediate_size = intermediate_size
			self.hidden_dropout_prob = hidden_dropout_prob
			self.attention_probs_dropout_prob = attention_probs_dropout_prob
			self.max_position_embeddings = max_position_embeddings
			self.type_vocab_size = type_vocab_size
			self.initializer_range = initializer_range
		else:
			raise ValueError("First argument must be either a vocabulary size (int)"
							 "or the path to a pretrained model config file (str)")

	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `BertConfig` from a Python dictionary of parameters."""
		config = BertConfig(vocab_size_or_config_json_file=-1)
		for key, value in json_object.items():
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
		with open(json_file, "r", encoding='utf-8') as reader:
			text = reader.read()
		fields = json.loads(text)
		# fields["hidden_size"] = 144
		# fields["type_vocab_size"] = 10000
		return cls.from_dict(fields)

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

try:
	from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
	print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
	class BertLayerNorm(nn.Module):
		def __init__(self, hidden_size, eps=1e-12):
			"""Construct a layernorm module in the TF style (epsilon inside the square root).
			"""
			super(BertLayerNorm, self).__init__()
			self.weight = nn.Parameter(torch.ones(hidden_size))
			self.bias = nn.Parameter(torch.zeros(hidden_size))
			self.variance_epsilon = eps

		def forward(self, x):
			u = x.mean(-1, keepdim=True)
			s = (x - u).pow(2).mean(-1, keepdim=True)
			x = (x - u) / torch.sqrt(s + self.variance_epsilon)
			return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
	"""Construct the embeddings from word, position and token_type embeddings.
	"""
	def __init__(self, config):
		super(BertEmbeddings, self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids, token_type_ids=None):
		seq_length = input_ids.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		words_embeddings = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(position_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = words_embeddings + position_embeddings + token_type_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings


class BertSelfAttention(nn.Module):
	def __init__(self, config):
		super(BertSelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer


class BertSelfOutput(nn.Module):
	def __init__(self, config):
		super(BertSelfOutput, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertAttention(nn.Module):
	def __init__(self, config):
		super(BertAttention, self).__init__()
		self.self = BertSelfAttention(config)
		self.output = BertSelfOutput(config)

	def forward(self, input_tensor, attention_mask):
		self_output = self.self(input_tensor, attention_mask)
		attention_output = self.output(self_output, input_tensor)
		return attention_output


class BertIntermediate(nn.Module):
	def __init__(self, config):
		super(BertIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		self.intermediate_act_fn = ACT2FN[config.hidden_act] \
			if isinstance(config.hidden_act, str) else config.hidden_act

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BertOutput(nn.Module):
	def __init__(self, config):
		super(BertOutput, self).__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertLayer(nn.Module):
	def __init__(self, config):
		super(BertLayer, self).__init__()
		self.attention = BertAttention(config)
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(self, hidden_states, attention_mask):
		attention_output = self.attention(hidden_states, attention_mask)
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output


class BertEncoder(nn.Module):
	def __init__(self, config):
		super(BertEncoder, self).__init__()
		layer = BertLayer(config)
		self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

	def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
		all_encoder_layers = []
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states, attention_mask)
			if output_all_encoded_layers:
				all_encoder_layers.append(hidden_states)
		if not output_all_encoded_layers:
			all_encoder_layers.append(hidden_states)
		return all_encoder_layers


class BertPooler(nn.Module):
	def __init__(self, config):
		super(BertPooler, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()

	def forward(self, hidden_states):
		# We "pool" the model by simply taking the hidden state corresponding
		# to the first token.
		first_token_tensor = hidden_states[:, 0]
		pooled_output = self.dense(first_token_tensor)
		pooled_output = self.activation(pooled_output)
		return pooled_output


class BertPredictionHeadTransform(nn.Module):
	def __init__(self, config):
		super(BertPredictionHeadTransform, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.transform_act_fn = ACT2FN[config.hidden_act] \
			if isinstance(config.hidden_act, str) else config.hidden_act
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.transform_act_fn(hidden_states)
		hidden_states = self.LayerNorm(hidden_states)
		return hidden_states


class BertLMPredictionHead(nn.Module):
	def __init__(self, config, bert_model_embedding_weights):
		super(BertLMPredictionHead, self).__init__()
		self.transform = BertPredictionHeadTransform(config)

		# The output weights are the same as the input embeddings, but there is
		# an output-only bias for each token.
		self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
								 bert_model_embedding_weights.size(0),
								 bias=False)
		self.decoder.weight = bert_model_embedding_weights
		self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

	def forward(self, hidden_states):
		hidden_states = self.transform(hidden_states)
		hidden_states = self.decoder(hidden_states) + self.bias
		return hidden_states


class BertOnlyMLMHead(nn.Module):
	def __init__(self, config, bert_model_embedding_weights):
		super(BertOnlyMLMHead, self).__init__()
		self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

	def forward(self, sequence_output):
		prediction_scores = self.predictions(sequence_output)
		return prediction_scores


class BertOnlyNSPHead(nn.Module):
	def __init__(self, config):
		super(BertOnlyNSPHead, self).__init__()
		self.seq_relationship = nn.Linear(config.hidden_size, 2)

	def forward(self, pooled_output):
		seq_relationship_score = self.seq_relationship(pooled_output)
		return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
	def __init__(self, config, bert_model_embedding_weights):
		super(BertPreTrainingHeads, self).__init__()
		self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
		self.seq_relationship = nn.Linear(config.hidden_size, 2)

	def forward(self, sequence_output, pooled_output):
		prediction_scores = self.predictions(sequence_output)
		seq_relationship_score = self.seq_relationship(pooled_output)
		return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
	""" An abstract class to handle weights initialization and
		a simple interface for dowloading and loading pretrained models.
	"""
	def __init__(self, config, *inputs, **kwargs):
		# import pdb; pdb.set_trace()
		# print("should have stoppppppppeeeeddddddddddd!!!!!...........................!!!!!!!")
		super(PreTrainedBertModel, self).__init__()
		if not isinstance(config, BertConfig):
			raise ValueError(
				"Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
				"To create a model from a Google pretrained model use "
				"`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
					self.__class__.__name__, self.__class__.__name__
				))
		self.config = config

	def init_bert_weights(self, module):
		""" Initialize the weights.
		"""
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			# module.weight.data.zero_()
		elif isinstance(module, BertLayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	@classmethod
	def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
		# import pdb; pdb.set_trace()
		# print("should have stoppppppppeeeeddddddddddd!!!!!...........................!!!!!!!")
		"""
		Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
		Download and cache the pre-trained model file if needed.

		Params:
			pretrained_model_name: either:
				- a str with the name of a pre-trained model to load selected in the list of:
					. `bert-base-uncased`
					. `bert-large-uncased`
					. `bert-base-cased`
					. `bert-large-cased`
					. `bert-base-multilingual-uncased`
					. `bert-base-multilingual-cased`
					. `bert-base-chinese`
				- a path or url to a pretrained model archive containing:
					. `bert_config.json` a configuration file for the model
					. `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
			cache_dir: an optional path to a folder in which the pre-trained models will be cached.
			state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
			*inputs, **kwargs: additional input for the specific Bert class
				(ex: num_labels for BertForSequenceClassification)
		"""
		# import pdb; pdb.set_trace()
		if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
			archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
		else:
			archive_file = pretrained_model_name
		# redirect to the cache, if necessary
		try:
			resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
		except FileNotFoundError:
			logger.error(
				"Model name '{}' was not found in model name list ({}). "
				"We assumed '{}' was a path or url but couldn't find any file "
				"associated to this path or url.".format(
					pretrained_model_name,
					', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
					archive_file))
			return None
		if resolved_archive_file == archive_file:
			logger.info("loading archive file {}".format(archive_file))
		else:
			logger.info("loading archive file {} from cache at {}".format(
				archive_file, resolved_archive_file))
		tempdir = None

		if os.path.isdir(resolved_archive_file):
			serialization_dir = resolved_archive_file
		else:
			# Extract archive to temp dir
			tempdir = tempfile.mkdtemp()
			logger.info("extracting archive file {} to temp dir {}".format(
				resolved_archive_file, tempdir))
			try:
				with tarfile.open(resolved_archive_file, 'r:gz') as archive:
	def is_within_directory(directory, target):
		
		abs_directory = os.path.abspath(directory)
		abs_target = os.path.abspath(target)
	
		prefix = os.path.commonprefix([abs_directory, abs_target])
		
		return prefix == abs_directory
	
	def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
	
		for member in tar.getmembers():
			member_path = os.path.join(path, member.name)
			if not is_within_directory(path, member_path):
				raise Exception("Attempted Path Traversal in Tar File")
	
		tar.extractall(path, members, numeric_owner=numeric_owner) 
		
	
	safe_extract(archive, tempdir)
			except:
				import pdb; pdb.set_trace()
				with tarfile.open(resolved_archive_file) as archive:
					with archive.open("scibert_scivocab_uncased/weights.tar.gz",  'r:gz') as archive2:
	def is_within_directory(directory, target):
		
		abs_directory = os.path.abspath(directory)
		abs_target = os.path.abspath(target)
	
		prefix = os.path.commonprefix([abs_directory, abs_target])
		
		return prefix == abs_directory
	
	def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
	
		for member in tar.getmembers():
			member_path = os.path.join(path, member.name)
			if not is_within_directory(path, member_path):
				raise Exception("Attempted Path Traversal in Tar File")
	
		tar.extractall(path, members, numeric_owner=numeric_owner) 
		
	
	safe_extract(archive2, tempdir)
			serialization_dir = tempdir
		# Load config
		print(os.path.join(serialization_dir, CONFIG_NAME))
		config_file = os.path.join(serialization_dir, CONFIG_NAME)
		config = BertConfig.from_json_file(config_file)
		logger.info("Model config {}".format(config))
		# Instantiate model.
		model = cls(config, *inputs, **kwargs)
		if state_dict is None:
			weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
			state_dict = torch.load(weights_path)

		old_keys = []
		new_keys = []
		for key in state_dict.keys():
			new_key = None
			if 'gamma' in key:
				new_key = key.replace('gamma', 'weight')
			if 'beta' in key:
				new_key = key.replace('beta', 'bias')
			if new_key:
				old_keys.append(key)
				new_keys.append(new_key)
		for old_key, new_key in zip(old_keys, new_keys):
			state_dict[new_key] = state_dict.pop(old_key)

		missing_keys = []
		unexpected_keys = []
		error_msgs = []
		# copy state_dict so _load_from_state_dict can modify it
		metadata = getattr(state_dict, '_metadata', None)
		state_dict = state_dict.copy()
		if metadata is not None:
			state_dict._metadata = metadata

		def load(module, prefix=''):
			local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
			module._load_from_state_dict(
				state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
			for name, child in module._modules.items():
				if child is not None:
					load(child, prefix + name + '.')
		load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
		if len(missing_keys) > 0:
			logger.info("Weights of {} not initialized from pretrained model: {}".format(
				model.__class__.__name__, missing_keys))
		if len(unexpected_keys) > 0:
			logger.info("Weights from pretrained model not used in {}: {}".format(
				model.__class__.__name__, unexpected_keys))
		if tempdir:
			# Clean up temp dir
			shutil.rmtree(tempdir)
		return model


class BertModel(PreTrainedBertModel):
	"""BERT model ("Bidirectional Embedding Representations from a Transformer").

	Params:
		config: a BertConfig class instance with the configuration to build a new model

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

	Outputs: Tuple of (encoded_layers, pooled_output)
		`encoded_layers`: controled by `output_all_encoded_layers` argument:
			- `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
				of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
				encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
			- `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
				to the last attention block of shape [batch_size, sequence_length, hidden_size],
		`pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
			classifier pretrained on top of the hidden state associated to the first character of the
			input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = modeling.BertModel(config=config)
	all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config):
		super(BertModel, self).__init__(config)
		self.embeddings = BertEmbeddings(config)
		self.encoder = BertEncoder(config)
		self.pooler = BertPooler(config)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		# We create a 3D attention mask from a 2D tensor mask.
		# Sizes are [batch_size, 1, 1, to_seq_length]
		# So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
		# this attention mask is more simple than the triangular masking of causal attention
		# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
		if attention_mask.dim() == 2:
			extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		else:
			extended_attention_mask = attention_mask.unsqueeze(1)

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		embedding_output = self.embeddings(input_ids, token_type_ids)
		encoded_layers = self.encoder(embedding_output,
									  extended_attention_mask,
									  output_all_encoded_layers=output_all_encoded_layers)
		sequence_output = encoded_layers[-1]
		pooled_output = self.pooler(sequence_output)
		if not output_all_encoded_layers:
			encoded_layers = encoded_layers[-1]
		return encoded_layers, pooled_output


class BertForPreTraining(PreTrainedBertModel):
	"""BERT model with pre-training heads.
	This module comprises the BERT model followed by the two pre-training heads:
		- the masked language modeling head, and
		- the next sentence classification head.

	Params:
		config: a BertConfig class instance with the configuration to build a new model.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
			with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
			is only computed for the labels set in [0, ..., vocab_size]
		`next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
			with indices selected in [0, 1].
			0 => next sentence is the continuation, 1 => next sentence is a random sentence.

	Outputs:
		if `masked_lm_labels` and `next_sentence_label` are not `None`:
			Outputs the total_loss which is the sum of the masked language modeling loss and the next
			sentence classification loss.
		if `masked_lm_labels` or `next_sentence_label` is `None`:
			Outputs a tuple comprising
			- the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
			- the next sentence classification logits of shape [batch_size, 2].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = BertForPreTraining(config)
	masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config):
		super(BertForPreTraining, self).__init__(config)
		self.bert = BertModel(config)
		self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
		sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
												   output_all_encoded_layers=False)
		prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

		if masked_lm_labels is not None and next_sentence_label is not None:
			loss_fct = CrossEntropyLoss(ignore_index=-1)
			masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
			next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
			total_loss = masked_lm_loss + next_sentence_loss
			return total_loss
		else:
			return prediction_scores, seq_relationship_score


class BertForMaskedLM(PreTrainedBertModel):
	"""BERT model with the masked language modeling head.
	This module comprises the BERT model followed by the masked language modeling head.

	Params:
		config: a BertConfig class instance with the configuration to build a new model.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
			with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
			is only computed for the labels set in [0, ..., vocab_size]

	Outputs:
		if `masked_lm_labels` is  not `None`:
			Outputs the masked language modeling loss.
		if `masked_lm_labels` is `None`:
			Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = BertForMaskedLM(config)
	masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config):
		super(BertForMaskedLM, self).__init__(config)
		self.bert = BertModel(config)
		self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
									   output_all_encoded_layers=False)
		prediction_scores = self.cls(sequence_output)

		if masked_lm_labels is not None:
			loss_fct = CrossEntropyLoss(ignore_index=-1)
			masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
			return masked_lm_loss
		else:
			return prediction_scores


class BertForNextSentencePrediction(PreTrainedBertModel):
	"""BERT model with next sentence prediction head.
	This module comprises the BERT model followed by the next sentence classification head.

	Params:
		config: a BertConfig class instance with the configuration to build a new model.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
			with indices selected in [0, 1].
			0 => next sentence is the continuation, 1 => next sentence is a random sentence.

	Outputs:
		if `next_sentence_label` is not `None`:
			Outputs the total_loss which is the sum of the masked language modeling loss and the next
			sentence classification loss.
		if `next_sentence_label` is `None`:
			Outputs the next sentence classification logits of shape [batch_size, 2].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = BertForNextSentencePrediction(config)
	seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config):
		super(BertForNextSentencePrediction, self).__init__(config)
		self.bert = BertModel(config)
		self.cls = BertOnlyNSPHead(config)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
									 output_all_encoded_layers=False)
		seq_relationship_score = self.cls( pooled_output)

		if next_sentence_label is not None:
			loss_fct = CrossEntropyLoss(ignore_index=-1)
			next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
			return next_sentence_loss
		else:
			return seq_relationship_score


class BertForSequenceClassification(PreTrainedBertModel):
	"""BERT model for classification.
	This module is composed of the BERT model with a linear layer on top of
	the pooled output.

	Params:
		`config`: a BertConfig class instance with the configuration to build a new model.
		`num_labels`: the number of classes for the classifier. Default = 2.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
			with indices selected in [0, ..., num_labels].

	Outputs:
		if `labels` is not `None`:
			Outputs the CrossEntropy classification loss of the output with the labels.
		if `labels` is `None`:
			Outputs the classification logits of shape [batch_size, num_labels].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	num_labels = 2

	model = BertForSequenceClassification(config, num_labels)
	logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, num_labels=2):
		super(BertForSequenceClassification, self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			return loss
		else:
			return logits


class BertForMultipleChoice(PreTrainedBertModel):
	"""BERT model for multiple choice tasks.
	This module is composed of the BERT model with a linear layer on top of
	the pooled output.

	Params:
		`config`: a BertConfig class instance with the configuration to build a new model.
		`num_choices`: the number of classes for the classifier. Default = 2.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
			with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
			and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
			with indices selected in [0, ..., num_choices].

	Outputs:
		if `labels` is not `None`:
			Outputs the CrossEntropy classification loss of the output with the labels.
		if `labels` is `None`:
			Outputs the classification logits of shape [batch_size, num_labels].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
	input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
	token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	num_choices = 2

	model = BertForMultipleChoice(config, num_choices)
	logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, num_choices=2):
		super(BertForMultipleChoice, self).__init__(config)
		self.num_choices = num_choices
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, 1)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		flat_input_ids = input_ids.view(-1, input_ids.size(-1))
		flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
		flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
		_, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		reshaped_logits = logits.view(-1, self.num_choices)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(reshaped_logits, labels)
			return loss
		else:
			return reshaped_logits


class BertForTokenClassification(PreTrainedBertModel):
	"""BERT model for token-level classification.
	This module is composed of the BERT model with a linear layer on top of
	the full hidden state of the last layer.

	Params:
		`config`: a BertConfig class instance with the configuration to build a new model.
		`num_labels`: the number of classes for the classifier. Default = 2.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
			with indices selected in [0, ..., num_labels].

	Outputs:
		if `labels` is not `None`:
			Outputs the CrossEntropy classification loss of the output with the labels.
		if `labels` is `None`:
			Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	num_labels = 2

	model = BertForTokenClassification(config, num_labels)
	logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, num_labels=2):
		super(BertForTokenClassification, self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		sequence_output = self.dropout(sequence_output)
		logits = self.classifier(sequence_output)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			return loss
		else:
			return logits


class BertForQuestionAnswering(PreTrainedBertModel):
	"""BERT model for Question Answering (span extraction).
	This module is composed of the BERT model with a linear layer on top of
	the sequence output that computes start_logits and end_logits

	Params:
		`config`: a BertConfig class instance with the configuration to build a new model.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
			Positions are clamped to the length of the sequence and position outside of the sequence are not taken
			into account for computing the loss.
		`end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
			Positions are clamped to the length of the sequence and position outside of the sequence are not taken
			into account for computing the loss.

	Outputs:
		if `start_positions` and `end_positions` are not `None`:
			Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
		if `start_positions` or `end_positions` is `None`:
			Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
			position tokens of shape [batch_size, sequence_length].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = BertForQuestionAnswering(config)
	start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config):
		super(BertForQuestionAnswering, self).__init__(config)
		# import pdb; pdb.set_trace()
		# print("should have stoppppppppeeeeddddddddddd!!!!!...........................!!!!!!!")
		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.qa_outputs = nn.Linear(config.hidden_size, 2)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,):
		# import pdb; pdb.set_trace()
		# print("should have stoppppppppeeeeddddddddddd!!!!!...........................!!!!!!!")
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)

		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)

			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2
			return total_loss
		else:
			return start_logits, end_logits


class BertForQuestionAnsweringWithClassifer(PreTrainedBertModel):
	"""BERT model for Question Answering (span extraction).
	This module is composed of the BERT model with a linear layer on top of
	the sequence output that computes start_logits and end_logits

	Params:
		`config`: a BertConfig class instance with the configuration to build a new model.

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see BERT paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
			Positions are clamped to the length of the sequence and position outside of the sequence are not taken
			into account for computing the loss.
		`end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
			Positions are clamped to the length of the sequence and position outside of the sequence are not taken
			into account for computing the loss.

	Outputs:
		if `start_positions` and `end_positions` are not `None`:
			Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
		if `start_positions` or `end_positions` is `None`:
			Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
			position tokens of shape [batch_size, sequence_length].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = BertForQuestionAnswering(config)
	start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, n_class=2, sequence_length=350):
		super(BertForQuestionAnsweringWithClassifer, self).__init__(config)
		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.qa_outputs = nn.Linear(config.hidden_size, 2)
		# self.qa_linear = nn.Linear()
		self.qa_classifier = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, answer_mask=None, known_switch=None):
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)
		switch_logits = self.qa_classifier(sequence_output.view(-1, 350*768))


		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			answer_mask = answer_mask.type(torch.FloatTensor).cuda()
			span_mask = (known_switch == 0).type(torch.FloatTensor).cuda()

			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)
			# [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions, span_mask)]
			loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)
			# import pdb; pdb.set_trace()
			# start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) for (_start_positions, _span_mask) in zip(torch.unbind(start_positions, dim=1), torch.unbind(span_mask, dim=1))]
			start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions, span_mask)]
			start_losses = [(loss_fct(start_logits[i].unsqueeze(0), start_pairs[i][0].unsqueeze(0)) * start_pairs[i][1]) for i in range(len(start_pairs))]
			
			end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions, span_mask)]
			end_losses = [(loss_fct(end_logits[i].unsqueeze(0), end_pairs[i][0].unsqueeze(0)) * end_pairs[i][1]) for i in range(len(end_pairs))]
			
			switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch, answer_mask)]
			# import pdb; pdb.set_trace()
			# print(switch_logits[0])
			# print(switch_pairs[0][0])
			switch_losses = [(loss_fct(switch_logits[i].unsqueeze(0), switch_pairs[i][0].unsqueeze(0))) for i in range(len(switch_pairs))]
			# end_loss = loss_fct(end_logits, end_positions)
			
			# end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
			#     for (_end_positions, _span_mask) \
			#     in zip(torch.unbind(end_positions, dim=1), torch.unbind(span_mask, dim=1))]
			# switch_losses = [(loss_fct(switch_logits, switch) * _answer_mask) \
			#     for (_switch, _answer_mask) \
			#     in zip(torch.unbind(switch, dim=1), torch.unbind(answer_mask, dim=1))]
			# start_loss = loss_fct(start_logits, start_positions)
			# end_loss = loss_fct(end_logits, end_positions)
			# switch_losses = loss_fct(switch_logits, switch)
			# import pdb; pdb.set_trace()
			# total_loss = (start_losses + end_losses + switch_losses) / 3
			total_loss = torch.sum(sum(start_losses + end_losses + switch_losses))
			return total_loss
		else:
			# import pdb; pdb.set_trace()
			return start_logits, end_logits, switch_logits




class BertForQuestionAnsweringWithClassifer_no_spans(PreTrainedBertModel):
	
	def __init__(self, config, n_class=2, sequence_length=350):
		super(BertForQuestionAnsweringWithClassifer_no_spans, self).__init__(config)
		self.bert = BertModel(config)
		
		self.qa_outputs = nn.Linear(config.hidden_size, 2)
		# self.qa_linear = nn.Linear()
		self.qa_classifier = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, known_switch=None):
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		switch_logits = self.qa_classifier(sequence_output.view(-1, 350*768))


		if known_switch is not None:
			

			span_mask = (known_switch == 0).type(torch.FloatTensor).cuda()

			# ignored_index = start_logits.size(1)
			loss_fct = CrossEntropyLoss(reduce=False)
			
			switch_losses = [(loss_fct(switch_logits[i].unsqueeze(0), known_switch[i].unsqueeze(0))) for i in range(len(known_switch))]

			total_loss = torch.sum(sum(switch_losses))
			return total_loss
		else:
			return start_logits, end_logits, switch_logits






class BertForQuestionAnsweringWithClassifer_LSTM(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=500, sequence_length=384):
		super(BertForQuestionAnsweringWithClassifer_LSTM, self).__init__(config)
		print(n_class)
		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		# self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(768, hidden_dim,bidirectional=True)
		self.hidden2tag = nn.Linear(hidden_dim*sequence_length*2, n_class)
		
		self.sequence_length = sequence_length

		self.qa_outputs = nn.Linear(config.hidden_size, 2)
		self.qa_classifier = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.hidden = self.init_hidden()
		
		self.apply(self.init_bert_weights)
		
		# self.hidden = self.hidden.cuda()

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(2, self.sequence_length, self.hidden_dim).cuda(),
				torch.zeros(2, self.sequence_length, self.hidden_dim).cuda())


	def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, answer_mask=None, known_switch=None):
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)
		# self.hidden = (self.hidden[0].type(torch.FloatTensor).cuda(), self.hidden[1].type(torch.FloatTensor).cuda()) 
		self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
		
		lstm_out, self.hidden = self.lstm(sequence_output, self.hidden)
		# import pdb; pdb.set_trace()

		switch_logits = self.hidden2tag(lstm_out.view(-1, 2*384*self.hidden_dim))
		# switch_logits = F.log_softmax(switch_space, dim=1)
		# switch_logits = self.qa_classifier(sequence_output.view(-1, 384*768))


		# print(switch_logits[0])
		# print (switch_logits[0][0] > switch_logits[0][1])

		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			answer_mask = answer_mask.type(torch.FloatTensor).cuda()
			span_mask = (known_switch == 0).type(torch.FloatTensor).cuda()

			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)
			loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

			start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions, span_mask)]
			start_losses = [(loss_fct(start_logits[i].unsqueeze(0), start_pairs[i][0].unsqueeze(0)) * start_pairs[i][1]) for i in range(len(start_pairs))]
			
			end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions, span_mask)]
			end_losses = [(loss_fct(end_logits[i].unsqueeze(0), end_pairs[i][0].unsqueeze(0)) * end_pairs[i][1]) for i in range(len(end_pairs))]
			
			switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch, answer_mask)]
			switch_losses = [(loss_fct(switch_logits[i].unsqueeze(0), switch_pairs[i][0].unsqueeze(0))) for i in range(len(switch_pairs))]
			total_loss = torch.sum(sum(start_losses + end_losses + switch_losses))
			# import pdb;pdb.set_trace()
			return total_loss
		else:
			# import pdb; pdb.set_trace()
			return start_logits, end_logits, switch_logits


class GlobalQuestionAnsweringWithLocationalPredictions(nn.Module):
	def __init__(self, locational_prediction_model,batch_size=2, n_class=3, hidden_dim=500, sequence_length=200):
		super(GlobalQuestionAnsweringWithLocationalPredictions, self).__init__()
		self.locational_prediction_model = locational_prediction_model
	 
		self.lstm_for_start= nn.LSTM(768, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.start_score_lstm = nn.Linear(hidden_dim*2, 1).cuda()
		self.hidden_start = self.init_hidden(batch_size)
		# self.lstm_for_end= nn.LSTM(768, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.end_score_lstm = nn.Linear(hidden_dim*2, 1).cuda()
		self.hidden_end = self.init_hidden(batch_size)
		self.lstm_for_start_before= nn.LSTM(768, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.start_score_lstm_before = nn.Linear(hidden_dim*2, 1).cuda()
		self.hidden_start_before = self.init_hidden(batch_size)
		# self.lstm_for_end_before= nn.LSTM(768, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.end_score_lstm_before = nn.Linear(hidden_dim*2, 1).cuda()
		self.hidden_end_before = self.init_hidden(batch_size)
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.sigmoid = nn.LogSigmoid()

	def init_hidden(self, batchsize):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(2, batchsize, 500).cuda(),
				torch.zeros(2, batchsize, 500).cuda())



	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None):
		# import pdb; pdb.set_trace()
		
		# if input_ids.size(0) != self.locational_prediction_model.hidden[0].size(1):
		#     self.locational_prediction_model.hidden = self.locational_prediction_model.init_hidden(len(input_ids))
		#     self.hidden_start = self.init_hidden(len(input_ids))
		#     self.hidden_start_before = self.init_hidden(len(input_ids))
		#     self.hidden_end = self.init_hidden(len(input_ids))
		#     self.hidden_end_before = self.init_hidden(len(input_ids))

# CUDA_VISIBLE_DEVICES=0 python3 global_local_predictions.py  --bert_model bert-base-uncased --do_train --do_predict   --do_lower_case   --train_file $SQUAD_DIR/train-gl-lc-v1.1.json   --predict_file $SQUAD_DIR/test-gl-lc-v1.1.json   --train_batch_size 2 --predict_batch_size 2  --learning_rate 3e-5   --num_train_epochs 20.0   --max_seq_length 200   --doc_stride 128   --output_dir /tmp/debug_squad_lstm_jointh300/
		if start_positions is not None and end_positions is not None: # training mode
			final_loss = torch.zeros(1).cuda()
			self.hidden_start = (Variable(self.hidden_start[0].data, requires_grad=True), Variable(self.hidden_start[1].data, requires_grad=True))
			self.hidden_start_before = (Variable(self.hidden_start_before[0].data, requires_grad=True), Variable(self.hidden_start_before[1].data, requires_grad=True))
			self.hidden_end = (Variable(self.hidden_end[0].data, requires_grad=True), Variable(self.hidden_end[1].data, requires_grad=True))
			self.hidden_end_before = (Variable(self.hidden_end_before[0].data, requires_grad=True), Variable(self.hidden_end_before[1].data, requires_grad=True))
			
			for i in range(len(input_ids[0])):
				# import pdb;pdb.set_trace()
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					continue
				new_loss, sequence_output = self.locational_prediction_model(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(),  start_positions[:,i].clone(), end_positions[:,i].clone(), answer_mask[:,i].clone(), known_switch[:,i].clone(), before_state_start_positions[:,i].clone(), before_state_end_positions[:,i].clone(),before_state_answer_mask[:,i].clone(),before_state_known_switch[:,i].clone())
				# import pdb; pdb.set_trace()
				# sequence_output = self.sigmoid(sequence_output)
				lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
				lstm_out = self.sigmoid(lstm_out)
				start_logits = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				# lstm_out, self.hidden_end = self.lstm_for_end(sequence_output, self.hidden_end)
				end_logits = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				lstm_out, self.hidden_start_before = self.lstm_for_start_before(sequence_output, self.hidden_start_before)
				lstm_out = self.sigmoid(lstm_out)
				before_state_start_logits = self.start_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				# lstm_out, self.hidden_end_before = self.lstm_for_end_before(sequence_output, self.hidden_end_before)
				before_state_end_logits = self.end_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				

				span_mask = (known_switch[:,i] == 0).type(torch.FloatTensor).cuda()
				before_state_span_mask = (before_state_known_switch[:,i] == 0).type(torch.FloatTensor).cuda()
				loss_fct = CrossEntropyLoss(reduce=False)

				start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions[:,i], span_mask)]
				# # import pdb; pdb.set_trace()
				start_losses = [(loss_fct(start_logits[j].unsqueeze(0), start_pairs[j][0].unsqueeze(0)) * start_pairs[j][1]) for j in range(len(start_pairs))]
				
				end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions[:,i], span_mask)]
				end_losses = [(loss_fct(end_logits[j].unsqueeze(0), end_pairs[j][0].unsqueeze(0)) * end_pairs[j][1]) for j in range(len(end_pairs))]


				before_state_start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(before_state_start_positions[:,i], before_state_span_mask)]
				before_state_start_losses = [(loss_fct(before_state_start_logits[j].unsqueeze(0), before_state_start_pairs[j][0].unsqueeze(0)) * before_state_start_pairs[j][1]) for j in range(len(before_state_start_pairs))]
			
				before_state_end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(before_state_end_positions[:,i], before_state_span_mask)]
				before_state_end_losses = [(loss_fct(before_state_end_logits[j].unsqueeze(0), before_state_end_pairs[j][0].unsqueeze(0)) * before_state_end_pairs[j][1]) for j in range(len(before_state_end_pairs))]
				
				# import pdb; pdb.set_trace()
				final_loss = torch.sum(sum(start_losses + end_losses + before_state_start_losses + before_state_end_losses )) + final_loss + new_loss
				# final_loss =final_loss +  new_loss + start_losses + end_losses + before_state_start_losses + before_state_end_losses
				# else:
				# final_loss = final_loss + self.locational_prediction_model(input_ids[:,i], token_type_ids[:,i], attention_mask[:,i],  start_positions[:,i], end_positions[:,i], answer_mask[:,i], known_switch[:,i], before_state_start_positions[:,i], before_state_end_positions[:,i],before_state_answer_mask[:,i],before_state_known_switch[:,i])
			return final_loss

		else:
			start_logits = []
			end_logits = []
			switch_logits = []
			before_state_start_logits = []
			before_state_end_logits = []
			before_state_switch_logits = []
			for i in range(len(input_ids[0])):
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					continue
				final_res_logits = self.locational_prediction_model(input_ids[:,i], token_type_ids[:,i], attention_mask[:,i])#,  start_positions[:,i], end_positions[:,i], answer_mask[:,i], known_switch[:,i], before_state_start_positions[:,i], before_state_end_positions[:,i],before_state_answer_mask[:,i],before_state_known_switch[:,i]))
				switch_logits.append(final_res_logits[0])
				before_state_switch_logits.append(final_res_logits[1])
				sequence_output = final_res_logits[2]
				# import pdb; pdb.set_trace()
				lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
				start_logit = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				# lstm_out, self.hidden_end = self.lstm_for_end(sequence_output, self.hidden_end)
				end_logit = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				lstm_out, self.hidden_start_before = self.lstm_for_start_before(sequence_output, self.hidden_start_before)
				before_state_start_logit = self.start_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				# lstm_out, self.hidden_end_before = self.lstm_for_end_before(sequence_output, self.hidden_end_before)
				before_state_end_logit = self.end_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)

				start_logits.append(start_logit)
				end_logits.append(end_logit)
				before_state_start_logits.append(before_state_start_logit)
				before_state_end_logits.append(before_state_end_logit)
			return start_logits, end_logits, switch_logits, before_state_start_logits, before_state_end_logits, before_state_switch_logits



class action_decoder(nn.Module):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
		super(action_decoder, self).__init__()
		self.batch_size = batch_size
		self.sequence_length = sequence_length
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.n_class = n_class
		



# for the ablations that the action sequences are not combination of the weighted bert represenatation.
class Sequence_decoder_simple(action_decoder):

	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
		super(Sequence_decoder_simple, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True, batch_first=True)
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * self.sequence_length, self.n_class)
		self.hidden = self.init_hidden(self.batch_size)
		
	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())


	def forward(self, input_ids, 
				sequence_list , start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None):
		
		
		if actions is not None:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0

				rnn_input = sequence_list[i]   # changed for list passing
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)

				action_logits = self.non_linear(rnn_output)
				action_logits = self.action_classifier(action_logits.view(-1, self.sequence_length*2*self.hidden_dim))
				loss_fct = CrossEntropyLoss(reduce=False)

				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				final_loss += torch.sum(sum(action_losses))
			return final_loss

		#evaluation			
		else:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

			action_logits_list = []
			for i in range(len(input_ids[0])):

				rnn_input = sequence_list[i]
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)

				action_logits = self.non_linear(rnn_output)
				action_logits = self.action_classifier(action_logits.view(-1, self.sequence_length*2*self.hidden_dim))

				action_logits_list.append(action_logits)
			return action_logits_list



class Sequence_decoder_no_lstm(action_decoder):

	def __init__(self, hidden_dim=300, input_dim= 768,  batch_size=8, n_class=4, sequence_length=200):
		super(Sequence_decoder_no_lstm, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(5* self.input_dim * self.sequence_length, self.n_class)
		self.hidden = self.init_hidden(self.batch_size)
		
	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())


	def forward(self, input_ids, sequence_list , start_probs=None, end_probs=None, classification_probs=None, 
							before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, 
							actions=None, unk_mask=None, none_mask=None, ans_mask=None):
		

		if actions is not None:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			
			final_loss = torch.zeros(1).cuda()
			before_state_start = None
			before_state_end =None
			padding_flag = [1 for x in range(self.batch_size)]
			
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

					# import pdb; pdb.set_trace()
					action_logits = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start_probs_with_attention, before_state_end_probs_with_attention), 2)
				
				else:
					action_logits = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start, before_state_end), 2)

				before_state_end = end_probs_with_attention
				before_state_start = start_probs_with_attention

				action_logits = self.action_classifier(action_logits.view(-1, self.sequence_length * 5 * self.input_dim))  # I change this so that the linear acts on the combination alone
				loss_fct = CrossEntropyLoss(reduce=False)


				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				final_loss += torch.sum(sum(action_losses))
			return final_loss


		else:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

			action_logits_list = []
			before_state_start = None
			before_state_end =None
			
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

					action_logits = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start_probs_with_attention, before_state_end_probs_with_attention), 2)
				
				else:
					action_logits = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start, before_state_end), 2)

				before_state_end = end_probs_with_attention
				before_state_start = start_probs_with_attention

				action_logits = self.action_classifier(action_logits.view(-1, self.sequence_length * 5 * self.input_dim))

				action_logits_list.append(action_logits)

			return action_logits_list


class CLS_decoder_sequential(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
		super(CLS_decoder_sequential, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		
		self.non_linear = nn.ReLU()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)

	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, sequence_list, start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None):
		
		self.batch_size = len(input_ids)
		if actions is not None:   #training
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			

			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0
				rnn_output, self.hidden = self.rnn(sequence_list[i][:,0].unsqueeze(1), self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				final_loss += torch.sum(sum(action_losses))
			
			return final_loss
		
		else:      #evaluation
			action_logits_list = []
			for i in range(len(input_ids[0])):
				
				rnn_output, self.hidden = self.rnn(sequence_list[i][:,0].unsqueeze(1), self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)
				action_logits_list.append(action_logits)
			return action_logits_list


class CLS_decoder_combined_represenatation(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
		super(CLS_decoder_combined_represenatation, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)
		self.previous_cls = torch.zeros(self.input_dim)

	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, sequence_list, start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None):
		
		self.batch_size = len(input_ids)
		if actions is not None:   #training
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			

			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			# import pdb; pdb.set_trace()
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0
				rnn_input = torch.cat((sequence_list[i][:,0].unsqueeze(1), self.previous_cls), 2)
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]
				self.previous_cls = sequence_list[i][:,0].unsqueeze(1)
				final_loss += torch.sum(sum(action_losses))
			
			return final_loss
		
		else:      #evaluation
			action_logits_list = []
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			for i in range(len(input_ids[0])):
				rnn_input = torch.cat((sequence_list[i][:,0].unsqueeze(1), self.previous_cls), 2)
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				self.previous_cls = sequence_list[i][:,0].unsqueeze(1)
				action_logits_list.append(action_logits)
			return action_logits_list



class CLS_decoder_seperate_berts_cls_per_sentence(PreTrainedBertModel):
	def __init__(self, config, n_class=8, hidden_dim=300, sequence_length=210, input_dim=768):
		super(CLS_decoder_seperate_berts_cls_per_sentence, self).__init__(config)
		self.batch_size = 8
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size
		self.input_dim = input_dim

		self.bert = BertModel(config)
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.input_dim*10 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)

	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  actions=None, cls_index_mask=None, weight_class_loss=1, weight_span_loss =1):
		
		self.batch_size = len(input_ids)
		if actions is not None:   #training
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			

			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

			for i in range(len(input_ids[0])):
				
				rnn_input = torch.zeros(self.batch_size, 15, 768).cuda()
				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)

				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0


				for j in range(len(input_ids)):
					batch_last_index = 0
					self.hidden = self.init_hidden(1)
					self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
					for ii in range(len(input_ids[j][i])):
						if cls_index_mask[j][i][ii] == 1:
							# import pdb; pdb.set_trace()
							rnn_input[j][batch_last_index] =sequence_output[:,ii][j].clone()
							batch_last_index += 1
				# rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				
				# action_logits = self.non_linear(rnn_input.squeeze())
				action_logits = self.action_classifier(rnn_input.view(-1,7680))

				# import pdb; pdb.set_trace()
				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]
				final_loss += torch.sum(sum(action_losses))
		
			return final_loss
		
		else:      #evaluation
			action_logits_list = []
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

			for i in range(len(input_ids[0])):
				
				rnn_input = torch.zeros(self.batch_size, 15, 768).cuda()
				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)

				for j in range(len(input_ids)):
					batch_last_index = 0
					self.hidden = self.init_hidden(1)
					self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
					for ii in range(len(input_ids[j][i])):
						if cls_index_mask[j][i][ii] == 1:
							# import pdb; pdb.set_trace()
							rnn_input[j][batch_last_index] =sequence_output[:,ii][j].clone()
							batch_last_index += 1
				# rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				
				# action_logits = self.non_linear(rnn_input.squeeze())
				action_logits = self.action_classifier(rnn_input.view(-1,7680))
				action_logits_list.append(action_logits)
			return action_logits_list
					


class CLS_decoder_cls_per_sentence(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=210):
		super(CLS_decoder_cls_per_sentence, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)
		self.previous_cls = torch.zeros(self.input_dim)

	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, sequence_list, start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None, cls_index_mask=None):
		
		self.batch_size = len(input_ids)
		if actions is not None:   #training
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			

			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0


				for j in range(len(input_ids)):
					self.hidden = self.init_hidden(1)
					self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
					for ii in range(len(input_ids[j][i])):
						if cls_index_mask[j][i][ii] == 1:
							rnn_input =sequence_list[i][:,ii][j].unsqueeze(0).unsqueeze(1)
							rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				
					action_logits = self.non_linear(rnn_output.squeeze())
					action_logits = self.action_classifier(action_logits)

					# import pdb; pdb.set_trace()
					loss_fct = CrossEntropyLoss(reduce=False)
					# action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
					action_losses = [(loss_fct(action_logits.unsqueeze(0), actions[j][i].unsqueeze(0)) * padding_flag[j])]
					final_loss += torch.sum(sum(action_losses))
			
			return final_loss
		
		else:      #evaluation
			action_logits_list = []
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			for i in range(len(input_ids[0])):
				final_action_logits = None
				for j in range(len(input_ids)):
					self.hidden = self.init_hidden(1)
					self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
					for ii in range(len(input_ids[j][i])):
						if cls_index_mask[j][i][ii] == 1:
							rnn_input =sequence_list[i][:,ii][j].unsqueeze(0).unsqueeze(1)
							rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				
					action_logits = self.non_linear(rnn_output.squeeze())
					action_logits = self.action_classifier(action_logits)
					if j == 0:
						final_action_logits = action_logits.unsqueeze(0)
					else:
						final_action_logits = torch.cat((final_action_logits, action_logits.unsqueeze(0)),0)
						
				action_logits_list.append(final_action_logits)
			return action_logits_list


class CLS_decoder_cls_per_sentence_concatination_sequential(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=210):
		super(CLS_decoder_cls_per_sentence_concatination_sequential, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)
		self.previous_cls = torch.zeros(self.input_dim)


	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, sequence_list, start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None, cls_index_mask=None):
		
		self.batch_size = len(input_ids)
		if actions is not None:   #training
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			

			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			# rnn_output = torch.zeros(self.batch_size, 1, 2*self.hidden_dim).cuda()
			# import pdb; pdb.set_trace()
			for i in range(len(input_ids[0])):
				
				rnn_input = torch.zeros(self.batch_size, 10, self.input_dim)

				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0


				for j in range(len(input_ids)):
					batch_last_index = 0
					self.hidden = self.init_hidden(1)
					self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
					for ii in range(len(input_ids[j][i])):
						if cls_index_mask[j][i][ii] == 1:
							# import pdb; pdb.set_trace()
							rnn_input[j][batch_last_index] =sequence_list[i][:,ii][j].clone()
							batch_last_index += 1
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				# import pdb; pdb.set_trace()
				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				self.previous_cls = sequence_list[i][:,0].unsqueeze(1)
				final_loss += torch.sum(sum(action_losses))
		
			return final_loss
		
		else:      #evaluation
			action_logits_list = []
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			for i in range(len(input_ids[0])):
				rnn_input = torch.cat((sequence_list[i][:,0].unsqueeze(1), self.previous_cls), 2)
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				self.previous_cls = sequence_list[i][:,0].unsqueeze(1)
				action_logits_list.append(action_logits)
			return action_logits_list



class CLS_decoder_cls_per_sentence_linear_decoder(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=210):
		super(CLS_decoder_cls_per_sentence_linear_decoder, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.input_dim*10 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)
		self.previous_cls = torch.zeros(self.input_dim)


	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, sequence_list, start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None, cls_index_mask=None):
		
		self.batch_size = len(input_ids)
		if actions is not None:   #training
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			

			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			# rnn_output = torch.zeros(self.batch_size, 1, 2*self.hidden_dim).cuda()
			# import pdb; pdb.set_trace()
			for i in range(len(input_ids[0])):
				
				rnn_input = torch.zeros(self.batch_size, 10, self.input_dim).cuda()

				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0


				for j in range(len(input_ids)):
					batch_last_index = 0
					self.hidden = self.init_hidden(1)
					self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
					for ii in range(len(input_ids[j][i])):
						if cls_index_mask[j][i][ii] == 1:
							if batch_last_index == 10:
								import pdb; pdb.set_trace()
							rnn_input[j][batch_last_index] =sequence_list[i][:,ii][j].clone()
							batch_last_index += 1
				# rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				# import pdb; pdb.set_trace()
				# action_logits = self.non_linear(rnn_input.squeeze())
				action_logits = self.action_classifier(rnn_input.view(-1,7680))

				# import pdb; pdb.set_trace()
				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				final_loss += torch.sum(sum(action_losses))
		
			return final_loss
		
		else:      #evaluation
			action_logits_list = []
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			for i in range(len(input_ids[0])):
				rnn_input = torch.cat((sequence_list[i][:,0].unsqueeze(1), self.previous_cls), 2)
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				self.previous_cls = sequence_list[i][:,0].unsqueeze(1)
				action_logits_list.append(action_logits)
			return action_logits_list


class CLS_decoder_cls_per_sentence_hirarchy_attn(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=210):
		super(CLS_decoder_cls_per_sentence_hirarchy_attn, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)
		self.previous_cls = torch.zeros(self.input_dim)


	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, sequence_list, start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None, cls_index_mask=None):
		
		self.batch_size = len(input_ids)
		if actions is not None:   #training
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			

			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			# rnn_output = torch.zeros(self.batch_size, 1, 2*self.hidden_dim).cuda()
			# import pdb; pdb.set_trace()
			for i in range(len(input_ids[0])):
				
				rnn_input = torch.zeros(self.batch_size, 1, self.input_dim).cuda()

				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0


				for j in range(len(input_ids)):
					self.hidden = self.init_hidden(1)
					self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
					ind = len(verts) - 1 - verts[::-1].index(value)
					
					rnn_input[j][0] =sequence_list[i][:,ind][j].clone()
				# rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				
				# action_logits = self.non_linear(rnn_input.squeeze())
				action_logits = self.action_classifier(action_logits)

				# import pdb; pdb.set_trace()
				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				final_loss += torch.sum(sum(action_losses))
		
			return final_loss
		
		else:      #evaluation
			action_logits_list = []
			self.previous_cls = torch.zeros(self.batch_size,1, self.input_dim).cuda()
			for i in range(len(input_ids[0])):
				rnn_input = torch.cat((sequence_list[i][:,0].unsqueeze(1), self.previous_cls), 2)
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				self.previous_cls = sequence_list[i][:,0].unsqueeze(1)
				action_logits_list.append(action_logits)
			return action_logits_list



class CLS_decoder_simple(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
		super(CLS_decoder_simple, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)

		#this classifier transfer a CLS token to action 768 --> 4 for propara. 
		self.action_classifier = nn.Linear(self.input_dim, n_class)

	def forward(self, input_ids, sequence_list, start_probs=None, end_probs=None, classification_probs=None, before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, actions=None, unk_mask=None, none_mask=None, ans_mask=None):
			self.batch_size = len(input_ids)
			if actions is not None:   #training
				final_loss = torch.zeros(1).cuda()
				padding_flag = [1 for x in range(self.batch_size)]
				for i in range(len(input_ids[0])):
					for j in range(len(input_ids)):
						if sum(input_ids[j][i]).cpu().numpy() == 0:
							padding_flag[j] = 0

					action_logits = self.action_classifier(sequence_list[i][:,0])
					loss_fct = CrossEntropyLoss(reduce=False)
					action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
					action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]
					final_loss += torch.sum(sum(action_losses))
				
				return final_loss
			
			else:      #evaluation
				action_logits_list = []
				for i in range(len(input_ids[0])):
					action_logits = self.action_classifier(sequence_list[i][:,0])
					action_logits_list.append(action_logits)

				return action_logits_list



class Sequence_decoder(action_decoder):

	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
		super(Sequence_decoder, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)

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
			before_state_end =None
			
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


######################################################### No cls masking

class Sequence_decoder_no_cls_masking(action_decoder):

	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=8, n_class=4, sequence_length=200):
		super(Sequence_decoder_no_cls_masking, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)

		self.rnn = nn.LSTM(self.input_dim * 5, self.hidden_dim, bidirectional=True, batch_first=True)
		self.non_linear = nn.LogSigmoid()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * self.sequence_length, self.n_class)
		self.hidden = self.init_hidden(self.batch_size)
		
	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())


	def forward(self, input_ids, sequence_list , start_probs=None, end_probs=None, classification_probs=None, 
							before_state_start_probs=None, before_state_end_probs=None, before_state_classification_probs=None, 
							actions=None, unk_mask=None, none_mask=None, ans_mask=None):
		
		unk_mask[:,:,0] = 1
		none_mask[:,:,0] = 1
		ans_mask[:,:,0] = 1
		# unk_mask[:,:,1] = 1
		# none_mask[:,:,1] = 1
		# ans_mask[:,:,1] = 1
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
				action_logits = self.action_classifier(action_logits.view(-1, self.sequence_length*2* self.hidden_dim))
				
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
			before_state_end =None
			
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
				action_logits = self.action_classifier(action_logits.view(-1, self.sequence_length*2* self.hidden_dim))

				action_logits_list.append(action_logits)
			return action_logits_list


####################################################################################################################################
class GlobalQuestionAnsweringWithLocationalPredictions_classes_cls_for_classes_no_lstm(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=1000, sequence_length=210):
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_cls_for_classes_no_lstm, self).__init__(config)
		self.batch_size = 8
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size

		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		
		self.lstm_for_tagging = nn.LSTM(config.hidden_size, self.hidden_dim, bidirectional=True,  batch_first=True)
		self.lstm_for_start= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.lstm_for_start_before= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
	
		self.hidden2tag = nn.Linear(hidden_dim*sequence_length*2, n_class)
		self.hidden2tag_before_state = nn.Linear(hidden_dim*sequence_length*2, n_class)

		self.qa_outputs = nn.Linear(config.hidden_size, 4)
		self.qa_classifier = nn.Linear(config.hidden_size*1, n_class)
		self.qa_classifier_before = nn.Linear(config.hidden_size*1, n_class)
	
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
		
		self.apply(self.init_bert_weights)
		

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
								before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None, weight_class_loss=1, weight_span_loss =1):
	


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
				

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)


				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits, before_state_start_logits, before_state_end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)
				before_state_start_logits = before_state_start_logits.squeeze(-1)
				before_state_end_logits = before_state_end_logits.squeeze(-1)

				switch_logits = self.qa_classifier(sequence_output[:,1])
				before_state_switch_logits = self.qa_classifier_before(sequence_output[:,1])


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
			return final_loss, start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list

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

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)
				
				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits, before_state_start_logits, before_state_end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)
				before_state_start_logits = before_state_start_logits.squeeze(-1)
				before_state_end_logits = before_state_end_logits.squeeze(-1)

				switch_logits = self.qa_classifier(sequence_output[:,1])
				before_state_switch_logits = self.qa_classifier_before(sequence_output[:,1])


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


			return start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list


######################################################################## cooking dataset.





class GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=1000, sequence_length=200):
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one, self).__init__(config)
		self.batch_size = 8
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size

		self.bert = BertModel(config)
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
		
		self.apply(self.init_bert_weights)
		

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
								before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None, weight_class_loss=1, weight_span_loss =1):
	


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
				

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
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
			return final_loss, start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list

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

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
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


			return start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list


######################################################################## cooking dataset.

#for location n_class == 2 (NONE and move)
#for shape n_class == 5 (None or destination shape)
#for composition n_class == 2 (chnage or unchanged)


class Sequence_decoder_cooking(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=2, n_class=2, sequence_length=350):
		super(Sequence_decoder_cooking, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)
		self.rnn = nn.LSTM(3840, hidden_dim, bidirectional=True, batch_first=True)
		# self.qa_future_outputs = nn.Linear(config.hidden_size, 2)
		self.non_linear = nn.ReLU()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * sequence_length, n_class)
		# self.qa_future_context_switch = nn.Linear(config.hidden_size*sequence_length, 2)
		self.hidden = self.init_hidden(self.batch_size)
		self.sequence_length = sequence_length
		
	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())


	def forward(self, input_ids, sequence_list , start_probs=None, end_probs=None, classification_probs=None, actions=None, unk_mask=None, ans_mask=None):
		
		# import pdb; pdb.set_trace()

		# weigthed_start = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list).size()
		if actions is not None:
			
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			final_loss = torch.zeros(1).cuda()
			before_state_start = None
			before_state_end =None
			padding_flag = [1 for x in range(self.batch_size)]
			# import pdb; pdb.set_trace()
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0

				# import pdb; pdb.set_trace()


				weigthed_start_ans = torch.mul((start_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				weigthed_start_unk = torch.mul((start_probs[i] * unk_mask[:,i].clone()), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				start_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) 

				weigthed_start_ans = torch.mul((end_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				weigthed_start_unk = torch.mul((end_probs[i] *unk_mask[:,i]).clone(), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				end_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i]) 
				
				if i == 0:
					before_state_start_probs_with_attention = torch.zeros(sequence_list[i].size()).cuda()
					before_state_end_probs_with_attention = torch.zeros(sequence_list[i].size()).cuda()
					rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start_probs_with_attention, before_state_end_probs_with_attention), 2)
				
				else:
					rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start, before_state_end), 2)

				before_state_end = end_probs_with_attention
				before_state_start = start_probs_with_attention
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)

				action_logits = self.non_linear(rnn_output)
				action_logits = self.action_classifier(action_logits.contiguous().view(-1, self.sequence_length*600))
				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				# final_loss += torch.sum(loss_fct(action_logits, actions[:,i].clone()))
				final_loss += torch.sum(sum(action_losses))
			return final_loss
		else:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

			action_logits_list = []
			before_state_start = None
			before_state_end =None
			for i in range(len(input_ids[0])):


				weigthed_start_ans = torch.mul((start_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				weigthed_start_unk = torch.mul((start_probs[i] * unk_mask[:,i].clone()), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				start_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i])

				weigthed_start_ans = torch.mul((end_probs[i] * ans_mask[:,i].clone()), classification_probs[i][:,0].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				weigthed_start_unk = torch.mul((end_probs[i] *unk_mask[:,i]).clone(), classification_probs[i][:,1].clone().unsqueeze(-1).expand(self.batch_size,self.sequence_length))
				end_probs_with_attention = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list[i]) +  torch.mul(weigthed_start_ans.unsqueeze(2), sequence_list[i])

				if i == 0:
					
					before_state_start_probs_with_attention = torch.zeros(sequence_list[i].size()).cuda()
					before_state_end_probs_with_attention = torch.zeros(sequence_list[i].size()).cuda()
					rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start_probs_with_attention, before_state_end_probs_with_attention), 2)
				
				else:
					rnn_input = torch.cat((sequence_list[i], start_probs_with_attention, end_probs_with_attention, before_state_start, before_state_end), 2)

				before_state_end = end_probs_with_attention
				before_state_start = start_probs_with_attention
				rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)

				action_logits = self.non_linear(rnn_output)
				action_logits = self.action_classifier(action_logits.contiguous().view(-1, self.sequence_length*600))


				action_logits_list.append(action_logits)
			return action_logits_list



#for location n_class == 243 (all spans + unk)
#for shape n_class == 5 (unk or shape)
#for composition n_class == 2 (chnage or unchanged)
class GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_cooking(PreTrainedBertModel):

	def __init__(self, config, n_class=243, hidden_dim=1000, sequence_length=350):
		# import pdb; pdb.set_trace()
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_cooking, self).__init__(config)
		self.batch_size = 2
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size

		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		
		self.lstm_for_tagging = nn.LSTM(config.hidden_size, self.hidden_dim, bidirectional=True,  batch_first=True)
		self.lstm_for_start= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.lstm_for_start_before= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		

		self.hidden2tag = nn.Linear(hidden_dim*sequence_length*2, n_class)
	
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
		
		self.apply(self.init_bert_weights)
		

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

	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None):
	
		if start_positions is not None and end_positions is not None: # training mode

			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []


			self.hidden = self.init_hidden(self.batch_size)
			# self.hidden_start_before = self.init_hidden(self.batch_size)
			self.hidden_start = self.init_hidden(self.batch_size)

			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			# self.hidden_start_before = (Variable(self.hidden_start_before[0].data, requires_grad=True), Variable(self.hidden_start_before[1].data, requires_grad=True))
			self.hidden_start = (Variable(self.hidden_start[0].data, requires_grad=True), Variable(self.hidden_start[1].data, requires_grad=True))
			
			self.prev_logits = None
			sequence_list = []
			# self.before_state_prev_logits = None
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0 
				

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)
				
				lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
				lstm_out = self.ReLU(lstm_out)
				start_logits = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				end_logits = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				
				# lstm_out_b, self.hidden_start_before = self.lstm_for_start_before(sequence_output, self.hidden_start_before)
				# lstm_out_b = self.ReLU(lstm_out_b)
				# before_state_start_logits = self.start_score_lstm_before(lstm_out_b.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				# before_state_end_logits = self.end_score_lstm_before(lstm_out_b.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)

				lstm_out_switch, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
				lstm_out_switch = self.ReLU(lstm_out_switch)
				switch_logits = self.hidden2tag(lstm_out_switch.contiguous().view(-1, 2*self.sequence_length*self.hidden_dim))
				# before_state_switch_logits = self.hidden2tag_before_state(lstm_out_switch.contiguous().view(-1, 2*self.sequence_length*self.hidden_dim))


				if len(start_positions[:,i].clone().size()) > 1:
					start_positions[:,i] = start_positions[:,i].clone().squeeze(-1)
				if len(end_positions[:,i].clone().size()) > 1:
					end_positions[:,i] = end_positions[:,i].clone().squeeze(-1)
				answer_mask[:,i] = answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
				span_mask = (known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()
				# before_state_answer_mask[:,i] = before_state_answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
				# before_state_span_mask = (before_state_known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()

				ignored_index = start_logits.size(1)
				start_positions[:,i].clone().clamp_(0, ignored_index)
				end_positions[:,i].clone().clamp_(0, ignored_index)

				loss_fct = CrossEntropyLoss(reduce=False)
				# import pdb; pdb.set_trace()
				start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions[:,i].clone(), span_mask)]
				start_losses = [(loss_fct(start_logits[ii].unsqueeze(0), start_pairs[ii][0].unsqueeze(0)) * start_pairs[ii][1] * padding_flag[ii]) for ii in range(len(start_pairs))]
				
				end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions[:,i].clone(), span_mask)]
				end_losses = [(loss_fct(end_logits[ii].unsqueeze(0), end_pairs[ii][0].unsqueeze(0)) * end_pairs[ii][1]* padding_flag[ii]) for ii in range(len(end_pairs))]
				
				switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch[:,i].clone(), answer_mask[:,i].clone())]
				switch_losses = [(loss_fct(switch_logits[ii].unsqueeze(0), switch_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(switch_pairs))]


				# before_state_start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(before_state_start_positions[:,i].clone(), before_state_span_mask)]
				# before_state_start_losses = [(loss_fct(before_state_start_logits[ii].unsqueeze(0), before_state_start_pairs[ii][0].unsqueeze(0)) * before_state_start_pairs[ii][1] * padding_flag[ii]) for ii in range(len(before_state_start_pairs))]
				
				# before_state_end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(before_state_end_positions[:,i].clone(), before_state_span_mask)]
				# before_state_end_losses = [(loss_fct(before_state_end_logits[ii].unsqueeze(0), before_state_end_pairs[ii][0].unsqueeze(0)) * before_state_end_pairs[ii][1]* padding_flag[ii]) for ii in range(len(before_state_end_pairs))]
				
				# before_state_switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(before_state_known_switch[:,i].clone(), before_state_answer_mask[:,i].clone())]
				# before_state_switch_losses = [(loss_fct(before_state_switch_logits[ii].unsqueeze(0), before_state_switch_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(before_state_switch_pairs))]
				total_loss = torch.sum(sum(start_losses + end_losses + switch_losses))
				
				start_probs = self.softmax(start_logits)		   
				end_probs = self.softmax(end_logits)
				switch_probs = self.softmax(switch_logits)
				# before_state_start_probs = self.softmax(before_state_start_logits)
				# before_state_end_probs = self.softmax(before_state_end_logits)
				# before_state_switch_probs = self.softmax(before_state_switch_logits)


				start_logits_list.append(start_probs)
				end_logits_list.append(end_probs)
				switch_logits_list.append(switch_probs)
				# before_state_start_logits_list.append(before_state_start_probs)
				# before_state_end_logits_list.append(before_state_end_probs)
				# before_state_switch_logits_list.append(before_state_switch_probs)

				final_loss += total_loss
			return final_loss, start_logits_list, end_logits_list, switch_logits_list, sequence_list

		else:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden_start_before = self.init_hidden(self.batch_size)
			self.hidden_start = self.init_hidden(self.batch_size)
			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []
			before_state_start_logits_list = []
			before_state_end_logits_list = []
			before_state_switch_logits_list = []
			sequence_list = []
			for i in range(len(input_ids[0])):
				
				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)

				lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
				lstm_out = self.ReLU(lstm_out)
				start_logits = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				end_logits = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				
				
				lstm_out_switch, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
				lstm_out_switch = self.ReLU(lstm_out_switch)
				switch_logits = self.hidden2tag(lstm_out_switch.contiguous().view(-1, 2*self.sequence_length*self.hidden_dim))


				start_probs = self.softmax(start_logits)		   
				end_probs = self.softmax(end_logits)
				switch_probs = self.softmax(switch_logits)
				

				start_logits_list.append(start_probs)
				end_logits_list.append(end_probs)
				switch_logits_list.append(switch_probs)
			

			return start_logits_list, end_logits_list, switch_logits_list, sequence_list



#################################################################################################################################################################

######################################################################## cooking dataset.

#for location n_class == 2 (NONE and move)
#for shape n_class == 5 (None or destination shape)
#for composition n_class == 2 (chnage or unchanged)

class Sequence_decoder_cls_cooking(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=2, n_class=2, sequence_length=350):
		super(Sequence_decoder_cls_cooking, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)

		#this classifier transfer a CLS token to action 768 --> n_class for propara. 
		self.action_classifier = nn.Linear(self.input_dim, self.n_class)
	
	def forward(self, input_ids, sequence_list , start_probs=None, end_probs=None, classification_probs=None, actions=None, unk_mask=None, ans_mask=None):
		
		# import pdb; pdb.set_trace()



		# weigthed_start = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list).size()
		if actions is not None:
			self.batch_size = len(input_ids)
			
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0

				action_logits = self.action_classifier(sequence_list[i][:,0])
				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]
				final_loss += torch.sum(sum(action_losses))
			
			return final_loss
			
		else:
			self.batch_size = len(input_ids)
			action_logits_list = []
			for i in range(len(input_ids[0])):
					
					action_logits = self.action_classifier(sequence_list[i][:,0])
					action_logits_list.append(action_logits)
			return action_logits_list


class Sequence_decoder_cls_cooking_sequential(action_decoder):
	def __init__(self, hidden_dim=300, input_dim=768,  batch_size=2, n_class=2, sequence_length=350):
		super(Sequence_decoder_cls_cooking_sequential, self).__init__(hidden_dim, input_dim, batch_size, n_class, sequence_length)

		#this classifier transfer a CLS token to action 768 --> n_class for propara. 
		self.non_linear = nn.ReLU()
		self.action_classifier = nn.Linear(self.hidden_dim*2 * 1, n_class) # since we are only looking at 1 CLS token
		self.rnn = nn.LSTM(self.input_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden = self.init_hidden(self.batch_size)

	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())
	
	def forward(self, input_ids, sequence_list , start_probs=None, end_probs=None, classification_probs=None, actions=None, unk_mask=None, ans_mask=None):
		
		# import pdb; pdb.set_trace()

		# weigthed_start = torch.mul(weigthed_start_unk.unsqueeze(2), sequence_list).size()
		if actions is not None:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))

			
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0

				rnn_output, self.hidden = self.rnn(sequence_list[i][:,0].unsqueeze(1), self.hidden)
				action_logits = self.non_linear(rnn_output.squeeze())
				action_logits = self.action_classifier(action_logits)

				loss_fct = CrossEntropyLoss(reduce=False)
				action_pairs = [(_actions) for (_actions) in zip(actions[:,i].clone())]
				action_losses = [(loss_fct(action_logits[ii].unsqueeze(0), action_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(action_pairs))]

				final_loss += torch.sum(sum(action_losses))
			
			return final_loss
		
		else:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
			action_logits_list = []
			for i in range(len(input_ids[0])):
					
					rnn_output, self.hidden = self.rnn(sequence_list[i][:,0].unsqueeze(1), self.hidden)
					action_logits = self.non_linear(rnn_output.squeeze())
					action_logits = self.action_classifier(action_logits)
					action_logits_list.append(action_logits)

			return action_logits_list


#for location n_class == 243 (all spans + unk)
#for shape n_class == 5 (unk or shape)
#for composition n_class == 2 (chnage or unchanged)
class GlobalQuestionAnsweringWithLocationalPredictions_classes_cls_for_classes_no_lstm_cooking(PreTrainedBertModel):

	def __init__(self, config, n_class=243, hidden_dim=1000, sequence_length=350):
		# import pdb; pdb.set_trace()
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_cls_for_classes_no_lstm_cooking, self).__init__(config)
		self.batch_size = 2
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size

		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		
		self.lstm_for_tagging = nn.LSTM(config.hidden_size, self.hidden_dim, bidirectional=True,  batch_first=True)
		self.lstm_for_start= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.lstm_for_start_before= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		
		self.qa_outputs = nn.Linear(config.hidden_size, 2)
		self.qa_classifier = nn.Linear(config.hidden_size*1, n_class)
		self.qa_classifier_before = nn.Linear(config.hidden_size*1, n_class)

		self.hidden2tag = nn.Linear(hidden_dim*sequence_length*2, n_class)
	
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
		
		self.apply(self.init_bert_weights)
		

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

	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None):
	
		if start_positions is not None and end_positions is not None: # training mode

			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []


			self.prev_logits = None
			sequence_list = []
			# self.before_state_prev_logits = None
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0 
				

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)


				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)

				switch_logits = self.qa_classifier(sequence_output[:,1])
				

				if len(start_positions[:,i].clone().size()) > 1:
					start_positions[:,i] = start_positions[:,i].clone().squeeze(-1)
				if len(end_positions[:,i].clone().size()) > 1:
					end_positions[:,i] = end_positions[:,i].clone().squeeze(-1)
				answer_mask[:,i] = answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
				span_mask = (known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()
				# before_state_answer_mask[:,i] = before_state_answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
				# before_state_span_mask = (before_state_known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()

				ignored_index = start_logits.size(1)
				start_positions[:,i].clone().clamp_(0, ignored_index)
				end_positions[:,i].clone().clamp_(0, ignored_index)

				loss_fct = CrossEntropyLoss(reduce=False)
				# import pdb; pdb.set_trace()
				start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions[:,i].clone(), span_mask)]
				start_losses = [(loss_fct(start_logits[ii].unsqueeze(0), start_pairs[ii][0].unsqueeze(0)) * start_pairs[ii][1] * padding_flag[ii]) for ii in range(len(start_pairs))]
				
				end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions[:,i].clone(), span_mask)]
				end_losses = [(loss_fct(end_logits[ii].unsqueeze(0), end_pairs[ii][0].unsqueeze(0)) * end_pairs[ii][1]* padding_flag[ii]) for ii in range(len(end_pairs))]
				
				switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch[:,i].clone(), answer_mask[:,i].clone())]
				switch_losses = [(loss_fct(switch_logits[ii].unsqueeze(0), switch_pairs[ii][0].unsqueeze(0)) * padding_flag[ii]) for ii in range(len(switch_pairs))]


				total_loss = torch.sum(sum(start_losses + end_losses + switch_losses))
				
				start_probs = self.softmax(start_logits)		   
				end_probs = self.softmax(end_logits)
				switch_probs = self.softmax(switch_logits)


				start_logits_list.append(start_probs)
				end_logits_list.append(end_probs)
				switch_logits_list.append(switch_probs)

				final_loss += total_loss
			return final_loss, start_logits_list, end_logits_list, switch_logits_list, sequence_list

		else:
			self.batch_size = len(input_ids)
			self.hidden = self.init_hidden(self.batch_size)
			self.hidden_start_before = self.init_hidden(self.batch_size)
			self.hidden_start = self.init_hidden(self.batch_size)
			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []
			before_state_start_logits_list = []
			before_state_end_logits_list = []
			before_state_switch_logits_list = []
			sequence_list = []
			for i in range(len(input_ids[0])):
				
				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)


				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)

				switch_logits = self.qa_classifier(sequence_output[:,1])


				start_probs = self.softmax(start_logits)		   
				end_probs = self.softmax(end_logits)
				switch_probs = self.softmax(switch_logits)
				

				start_logits_list.append(start_probs)
				end_logits_list.append(end_probs)
				switch_logits_list.append(switch_probs)
			

			return start_logits_list, end_logits_list, switch_logits_list, sequence_list


#################################################################################################################################################################

class GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_no_lstm(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=1000, sequence_length=210):
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_no_lstm, self).__init__(config)
		self.batch_size = 8
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size

		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version

		self.qa_outputs = nn.Linear(config.hidden_size, 4)
		self.qa_classifier = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.qa_classifier_before = nn.Linear(config.hidden_size*sequence_length, n_class)
		#######################################

		self.softmax = nn.Softmax()
		self.apply(self.init_bert_weights)

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size

	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())


	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, 
							answer_mask=None, known_switch=None,
							before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None,weight_class_loss=1, weight_span_loss =1):
		
		if start_positions is not None and end_positions is not None: # training mode

			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []
			before_state_start_logits_list = []
			before_state_end_logits_list = []
			before_state_switch_logits_list = []
			sequence_list = []
			
			final_loss = torch.zeros(1).cuda()
			padding_flag = [1 for x in range(self.batch_size)]
			for i in range(len(input_ids[0])):
				for j in range(len(input_ids)):
					if sum(input_ids[j][i]).cpu().numpy() == 0:
						padding_flag[j] = 0 

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)
				
				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits, before_state_start_logits, before_state_end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)
				before_state_start_logits = before_state_start_logits.squeeze(-1)
				before_state_end_logits = before_state_end_logits.squeeze(-1)

				switch_logits = self.qa_classifier(sequence_output.view(-1, self.sequence_length*self.bert_hidden_size))
				before_state_switch_logits = self.qa_classifier_before(sequence_output.view(-1, self.sequence_length*self.bert_hidden_size))


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
				total_loss = torch.sum(sum(start_losses + end_losses + switch_losses + before_state_start_losses + before_state_end_losses + before_state_switch_losses))
				
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
			return final_loss, start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list

		else:
			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []
			before_state_start_logits_list = []
			before_state_end_logits_list = []
			before_state_switch_logits_list = []
			sequence_list = []
			for i in range(len(input_ids[0])):

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_list.append(sequence_output)

				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits, before_state_start_logits, before_state_end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)
				before_state_start_logits = before_state_start_logits.squeeze(-1)
				before_state_end_logits = before_state_end_logits.squeeze(-1)

				switch_logits = self.qa_classifier(sequence_output.view(-1, self.sequence_length*self.bert_hidden_size))
				before_state_switch_logits = self.qa_classifier_before(sequence_output.view(-1, self.sequence_length*self.bert_hidden_size))

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


			return start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_list



class GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_no_classifier(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=1000, sequence_length=200):
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_no_classifier, self).__init__(config)
		self.batch_size = 2
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size

		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		
		self.lstm_for_start= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
		self.lstm_for_start_before= nn.LSTM(config.hidden_size, hidden_dim,bidirectional=True,  batch_first=True).cuda()
	
		########################################

		self.start_score_lstm = nn.Linear(hidden_dim*2, 1).cuda()
		self.end_score_lstm = nn.Linear(hidden_dim*2, 1).cuda()
		self.start_score_lstm_before = nn.Linear(hidden_dim*2, 1).cuda()
		self.end_score_lstm_before = nn.Linear(hidden_dim*2, 1).cuda()
		
		self.sigmoid = nn.LogSigmoid()
		self.softmax = nn.Softmax()

		self.hidden_start = self.init_hidden(self.batch_size)
		self.hidden_start_before = self.init_hidden(self.batch_size)
		
		self.apply(self.init_bert_weights)
		

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.hidden_start = self.init_hidden(self.batch_size)
		self.hidden_start_before = self.init_hidden(self.batch_size)
		self.hidden = self.init_hidden(self.batch_size)


	def init_hidden(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())

	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None):

		if start_positions is not None and end_positions is not None: # training mode

			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []
			before_state_start_logits_list = []
			before_state_end_logits_list = []
			before_state_switch_logits_list = []

			self.hidden_start_before = (Variable(self.hidden_start_before[0].data, requires_grad=True), Variable(self.hidden_start_before[1].data, requires_grad=True))
			self.hidden_start = (Variable(self.hidden_start[0].data, requires_grad=True), Variable(self.hidden_start[1].data, requires_grad=True))
			self.hidden_spans_before = (Variable(self.hidden_spans_before[0].data, requires_grad=True), Variable(self.hidden_spans_before[1].data, requires_grad=True))

			final_loss = torch.zeros(1).cuda()
			for i in range(len(input_ids[0])):
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					tmp = torch.zeros(self.batch_size, self.bert_hidden_size).cuda()
					start_logits_list.append(tmp)
					end_logits_list.append(tmp)
					switch_logits_list.append(tmp)
					before_state_start_logits_list.append(tmp)
					before_state_end_logits_list.append(tmp)
					before_state_switch_logits_list.append(tmp)
					continue
				
				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_output = self.sigmoid(sequence_output)
				lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
				start_logits = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				end_logits = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				lstm_out, self.hidden_start_before = self.lstm_for_start_before(sequence_output, self.hidden_start_before)
				before_state_start_logits = self.start_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				before_state_end_logits = self.end_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)

				# lstm_out, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
				# switch_logits = self.hidden2tag(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))
				# before_state_switch_logits = self.hidden2tag_before_state(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))

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
				start_losses = [(loss_fct(start_logits[ii].unsqueeze(0), start_pairs[ii][0].unsqueeze(0)) * start_pairs[ii][1]) for ii in range(len(start_pairs))]
				
				end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions[:,i].clone(), span_mask)]
				end_losses = [(loss_fct(end_logits[ii].unsqueeze(0), end_pairs[ii][0].unsqueeze(0)) * end_pairs[ii][1]) for ii in range(len(end_pairs))]

				before_state_start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(before_state_start_positions[:,i].clone(), before_state_span_mask)]
				before_state_start_losses = [(loss_fct(before_state_start_logits[ii].unsqueeze(0), before_state_start_pairs[ii][0].unsqueeze(0)) * before_state_start_pairs[ii][1]) for ii in range(len(before_state_start_pairs))]
				
				before_state_end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(before_state_end_positions[:,i].clone(), before_state_span_mask)]
				before_state_end_losses = [(loss_fct(before_state_end_logits[ii].unsqueeze(0), before_state_end_pairs[ii][0].unsqueeze(0)) * before_state_end_pairs[ii][1]) for ii in range(len(before_state_end_pairs))]
				
				total_loss = torch.sum(sum(start_losses + end_losses  + before_state_start_losses + before_state_end_losses))
				
				start_probs = self.softmax(start_logits)		   
				end_probs = self.softmax(end_logits)
				before_state_start_probs = self.softmax(before_state_start_logits)
				before_state_end_probs = self.softmax(before_state_end_logits)


				start_logits_list.append(start_probs)
				end_logits_list.append(end_probs)
				switch_logits_list.append(torch.ones(2,3).cuda())
				before_state_start_logits_list.append(before_state_start_probs)
				before_state_end_logits_list.append(before_state_end_probs)
				before_state_switch_logits_list.append(torch.ones(2,3).cuda())

				final_loss += total_loss
			return final_loss, start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_output

		else:
			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []
			before_state_start_logits_list = []
			before_state_end_logits_list = []
			before_state_switch_logits_list = []
			for i in range(len(input_ids[0])):
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					continue
				

				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				sequence_output = self.sigmoid(sequence_output)
				lstm_out, self.hidden_start = self.lstm_for_start(sequence_output, self.hidden_start)
				start_logits = self.start_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				end_logits = self.end_score_lstm(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				lstm_out, self.hidden_start_before = self.lstm_for_start_before(sequence_output, self.hidden_start_before)
				before_state_start_logits = self.start_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)
				before_state_end_logits = self.end_score_lstm_before(lstm_out.contiguous().view(-1,self.sequence_length, 2*self.hidden_dim)).squeeze(-1)

				start_probs = self.softmax(start_logits)		   
				end_probs = self.softmax(end_logits)
				before_state_start_probs = self.softmax(before_state_start_logits)
				before_state_end_probs = self.softmax(before_state_end_logits)


				start_logits_list.append(start_probs)
				end_logits_list.append(end_probs)
				switch_logits_list.append(torch.ones(2,3).cuda())
				before_state_start_logits_list.append(before_state_start_probs)
				before_state_end_logits_list.append(before_state_end_probs)
				before_state_switch_logits_list.append(torch.ones(2,3).cuda())

			return start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list, sequence_output


class GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_lstm_only_on_classification(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=500, sequence_length=200):
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_only_all_in_one_lstm_only_on_classification, self).__init__(config)
		self.qa_outputs = nn.Linear(config.hidden_size, 4)
		self.batch_size = 2
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length
		self.bert_hidden_size = config.hidden_size

		self.bert = BertModel(config)
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

		self.hidden_end = self.init_hidden(self.batch_size)
		self.hidden_start = self.init_hidden(self.batch_size)
		self.hidden_start_before = self.init_hidden(self.batch_size)
		self.hidden_end_before = self.init_hidden(self.batch_size)
		self.hidden = self.init_hidden(self.batch_size)
		
		self.apply(self.init_bert_weights)
		

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


	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None):

		self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
		
		if start_positions is not None and end_positions is not None: # training mode
			final_loss = torch.zeros(1).cuda()
			for i in range(len(input_ids[0])):
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					continue
				sequence_output, _ = self.bert(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(), output_all_encoded_layers=False)
				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits, before_state_start_logits, before_state_end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)
				before_state_start_logits = before_state_start_logits.squeeze(-1)
				before_state_end_logits = before_state_end_logits.squeeze(-1)

				lstm_out, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
				switch_logits = self.hidden2tag(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))
				before_state_switch_logits = self.hidden2tag_before_state(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))
				
				if len(start_positions[:,i].clone().size()) > 1:
					start_positions[:,i] = start_positions[:,i].clone().squeeze(-1)
				if len(end_positions[:,i].clone().size()) > 1:
					end_positions[:,i] = end_positions[:,i].clone().squeeze(-1)
				answer_mask[:,i] = answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
				span_mask = (known_switch[:,i] == 0).type(torch.FloatTensor).cuda()
				before_state_answer_mask[:,i] = before_state_answer_mask[:,i].clone().type(torch.FloatTensor).cuda()
				before_state_span_mask = (before_state_known_switch[:,i].clone() == 0).type(torch.FloatTensor).cuda()

				ignored_index = start_logits.size(1)
				start_positions[:,i].clone().clamp_(0, ignored_index)
				end_positions[:,i].clone().clamp_(0, ignored_index)

				loss_fct = CrossEntropyLoss(reduce=False)
				start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions[:,i].clone(), span_mask)]
				start_losses = [(loss_fct(start_logits[ii].unsqueeze(0), start_pairs[ii][0].unsqueeze(0)) * start_pairs[ii][1]) for ii in range(len(start_pairs))]
				
				end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions[:,i].clone(), span_mask)]
				end_losses = [(loss_fct(end_logits[ii].unsqueeze(0), end_pairs[ii][0].unsqueeze(0)) * end_pairs[ii][1]) for ii in range(len(end_pairs))]
				
				switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch[:,i].clone(), answer_mask[:,i].clone())]
				switch_losses = [(loss_fct(switch_logits[ii].unsqueeze(0), switch_pairs[ii][0].unsqueeze(0))) for ii in range(len(switch_pairs))]


				before_state_start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(before_state_start_positions[:,i].clone(), before_state_span_mask)]
				before_state_start_losses = [(loss_fct(before_state_start_logits[ii].unsqueeze(0), before_state_start_pairs[ii][0].unsqueeze(0)) * before_state_start_pairs[ii][1]) for ii in range(len(before_state_start_pairs))]
				
				before_state_end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(before_state_end_positions[:,i].clone(), before_state_span_mask)]
				before_state_end_losses = [(loss_fct(before_state_end_logits[ii].unsqueeze(0), before_state_end_pairs[ii][0].unsqueeze(0)) * before_state_end_pairs[ii][1]) for ii in range(len(before_state_end_pairs))]
				
				before_state_switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(before_state_known_switch[:,i].clone(), before_state_answer_mask[:,i].clone())]
				before_state_switch_losses = [(loss_fct(before_state_switch_logits[ii].unsqueeze(0), before_state_switch_pairs[ii][0].unsqueeze(0))) for ii in range(len(before_state_switch_pairs))]
				total_loss = torch.sum(sum(start_losses + end_losses + switch_losses + before_state_start_losses + before_state_end_losses + before_state_switch_losses))
			   

				final_loss = final_loss + total_loss
			return final_loss

		else:

			start_logits_list = []
			end_logits_list = []
			switch_logits_list = []
			before_state_start_logits_list = []
			before_state_end_logits_list = []
			before_state_switch_logits_list = []
			for i in range(len(input_ids[0])):
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					continue
				
				sequence_output, _ = self.bert(input_ids[:,i], token_type_ids[:,i], attention_mask[:,i], output_all_encoded_layers=False)
				logits = self.qa_outputs(sequence_output)
				start_logits, end_logits, before_state_start_logits, before_state_end_logits = logits.split(1, dim=-1)
				start_logits = start_logits.squeeze(-1)
				end_logits = end_logits.squeeze(-1)
				before_state_start_logits = before_state_start_logits.squeeze(-1)
				before_state_end_logits = before_state_end_logits.squeeze(-1)
				lstm_out, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
				switch_logits = self.hidden2tag(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))
				before_state_switch_logits = self.hidden2tag_before_state(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))

				start_logits_list.append(start_logits)
				end_logits_list.append(end_logits)
				switch_logits_list.append(switch_logits)
				before_state_start_logits_list.append(before_state_start_logits)
				before_state_end_logits_list.append(before_state_end_logits)
				before_state_switch_logits_list.append(before_state_switch_logits)
			return start_logits_list, end_logits_list, switch_logits_list, before_state_start_logits_list, before_state_end_logits_list, before_state_switch_logits_list




class GlobalQuestionAnsweringWithLocationalPredictions_classes_only(nn.Module):
	def __init__(self, locational_prediction_model,batch_size=2, n_class=3, hidden_dim=500, sequence_length=200):
		super(GlobalQuestionAnsweringWithLocationalPredictions_classes_only, self).__init__()
		self.locational_prediction_model = locational_prediction_model
		self.hidden_dim = hidden_dim
		self.sequence_length = sequence_length

	def init_hidden(self, batchsize):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(2, batchsize, 500).cuda(),
				torch.zeros(2, batchsize, 500).cuda())



	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None):

		if start_positions is not None and end_positions is not None: # training mode
			final_loss = torch.zeros(1).cuda()
			
			for i in range(len(input_ids[0])):
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					continue
				# final_loss = final_loss + self.locational_prediction_model(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(),  start_positions[:,i].clone(), end_positions[:,i].clone(), answer_mask[:,i].clone(), known_switch[:,i].clone(), before_state_start_positions[:,i].clone(), before_state_end_positions[:,i].clone(),before_state_answer_mask[:,i].clone(),before_state_known_switch[:,i].clone())
				final_loss = final_loss + self.locational_prediction_model(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone(),  start_positions[:,i].clone(), end_positions[:,i].clone(), answer_mask[:,i].clone(), known_switch[:,i].clone(), before_state_start_positions[:,i].clone(), before_state_end_positions[:,i].clone(),before_state_answer_mask[:,i].clone(),before_state_known_switch[:,i].clone())
			return final_loss

		else:
			start_logits = []
			end_logits = []
			switch_logits = []
			before_state_start_logits = []
			before_state_end_logits = []
			before_state_switch_logits = []
			for i in range(len(input_ids[0])):
				if sum(input_ids[0][i]).cpu().numpy() == 0:
					continue
				final_res_logits = self.locational_prediction_model(input_ids[:,i].clone(), token_type_ids[:,i].clone(), attention_mask[:,i].clone())#,  start_positions[:,i], end_positions[:,i], answer_mask[:,i], known_switch[:,i], before_state_start_positions[:,i], before_state_end_positions[:,i],before_state_answer_mask[:,i],before_state_known_switch[:,i]))
				start_logits.append(final_res_logits[0])
				end_logits.append(final_res_logits[1])
				switch_logits.append(final_res_logits[2])
				before_state_start_logits.append(final_res_logits[3])
				before_state_end_logits.append(final_res_logits[4])
				before_state_switch_logits.append(final_res_logits[5])
			return start_logits, end_logits, switch_logits, before_state_start_logits, before_state_end_logits, before_state_switch_logits






class BertForQuestionAnswering_locational(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=500, sequence_length=200):
		super(BertForQuestionAnswering_locational, self).__init__(config)
		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		# self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.hidden_dim = hidden_dim
		self.lstm_for_tagging = nn.LSTM(768, hidden_dim,bidirectional=True,  batch_first=True)
		self.hidden2tag = nn.Linear(hidden_dim*sequence_length*2, n_class)
		self.hidden2tag_before_state = nn.Linear(hidden_dim*sequence_length*2, n_class)
		
		self.sequence_length = sequence_length

		self.qa_outputs = nn.Linear(config.hidden_size, 4)
		self.qa_classifier = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.hidden = self.init_hidden(2)
		
		self.apply(self.init_bert_weights)
		
		# self.hidden = self.hidden.cuda()

	def init_hidden(self, batchsize):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())


	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None):
		# if input_ids.size(0) != self.hidden[0].size(1):
		#     self.hidden = self.init_hidden(input_ids.size(0))
		# import pdb; pdb.set_trace()
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		logits = self.qa_outputs(sequence_output)
		self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
		lstm_out, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)

		switch_logits = self.hidden2tag(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))
		before_state_switch_logits = self.hidden2tag_before_state(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))

		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)

			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			answer_mask = answer_mask.type(torch.FloatTensor).cuda()
			span_mask = (known_switch == 0).type(torch.FloatTensor).cuda()
			before_state_answer_mask = before_state_answer_mask.type(torch.FloatTensor).cuda()
			before_state_span_mask = (before_state_known_switch == 0).type(torch.FloatTensor).cuda()

			loss_fct = CrossEntropyLoss(reduce=False)

			
			switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch, answer_mask)]
			switch_losses = [(loss_fct(switch_logits[i].unsqueeze(0), switch_pairs[i][0].unsqueeze(0))) for i in range(len(switch_pairs))]


			
			before_state_switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(before_state_known_switch, before_state_answer_mask)]
			before_state_switch_losses = [(loss_fct(before_state_switch_logits[i].unsqueeze(0), before_state_switch_pairs[i][0].unsqueeze(0))) for i in range(len(before_state_switch_pairs))]

			# total_loss = torch.sum(sum(start_losses + end_losses + switch_losses + before_state_start_losses + before_state_end_losses + before_state_switch_losses))
			total_loss = torch.sum(sum(switch_losses + before_state_switch_losses))
			# import pdb;pdb.set_trace()
			return (total_loss, sequence_output)
		else:
			# import pdb; pdb.set_trace()
			return switch_logits, before_state_switch_logits, sequence_output


class BertForQuestionAnsweringWithClassifer_LSTM_before_state(PreTrainedBertModel):

	def __init__(self, config, n_class=3, hidden_dim=500, sequence_length=200):
		super(BertForQuestionAnsweringWithClassifer_LSTM_before_state, self).__init__(config)
		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		# self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.hidden_dim = hidden_dim
		self.lstm_for_tagging = nn.LSTM(768, hidden_dim,bidirectional=True,  batch_first=True)
		self.hidden2tag = nn.Linear(hidden_dim*sequence_length*2, n_class)
		self.hidden2tag_before_state = nn.Linear(hidden_dim*sequence_length*2, n_class)
		
		self.sequence_length = sequence_length

		self.qa_outputs = nn.Linear(config.hidden_size, 4)
		self.qa_classifier = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.qa_classifier_before = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.hidden = self.init_hidden1(8)
		
		self.apply(self.init_bert_weights)
		
		# self.hidden = self.hidden.cuda()

	def init_hidden1(self, batchsize):
		return (torch.zeros(2, batchsize, self.hidden_dim).cuda(),
				torch.zeros(2, batchsize, self.hidden_dim).cuda())


	def forward(self, input_ids, token_type_ids=None, attention_mask=None,  start_positions=None, end_positions=None, answer_mask=None, known_switch=None, before_state_start_positions=None, before_state_end_positions=None,before_state_answer_mask=None,before_state_known_switch=None):
		# import pdb; pdb.set_trace()
		# import pdb; pdb.set_trace()
		# print ("jshlkahsdlkajkldjalskdhajhdalkjs;aj;lsjkdal;skdlajs;ldajskdhajsbdjhavshdygaiusyddajlskjdaksndbhagdsyaidujailskd")
		# print (self.hidden[0].size())
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits, before_state_start_logits, before_state_end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)
		before_state_start_logits = before_state_start_logits.squeeze(-1)
		before_state_end_logits = before_state_end_logits.squeeze(-1)
		# self.hidden = (self.hidden[0].type(torch.FloatTensor).cuda(), self.hidden[1].type(torch.FloatTensor).cuda()) 
		# self.hidden = (Variable(self.hidden[0].data, requires_grad=True), Variable(self.hidden[1].data, requires_grad=True))
		# import pdb; pdb.set_trace()
		# lstm_out, self.hidden = self.lstm_for_tagging(sequence_output, self.hidden)
		# import pdb; pdb.set_trace()

		# switch_logits = self.hidden2tag(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))
		# before_state_switch_logits = self.hidden2tag_before_state(lstm_out.contiguous().view(-1, 2*200*self.hidden_dim))
		
		# switch_logits = F.log_softmax(switch_space, dim=1)
		
		switch_logits = self.qa_classifier(sequence_output.view(-1, 200*768))
		before_state_switch_logits = self.qa_classifier_before(sequence_output.view(-1, 200*768))


		# print(switch_logits[0])
		# print (switch_logits[0][0] > switch_logits[0][1])

		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)

			if len(before_state_start_positions.size()) > 1:
				before_state_start_positions = before_state_start_positions.squeeze(-1)
			if len(before_state_end_positions.size()) >  1:
				before_state_end_positions = before_state_end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			answer_mask = answer_mask.type(torch.FloatTensor).cuda()
			span_mask = (known_switch == 0).type(torch.FloatTensor).cuda()
			before_state_answer_mask = before_state_answer_mask.type(torch.FloatTensor).cuda()
			before_state_span_mask = (before_state_known_switch == 0).type(torch.FloatTensor).cuda()

			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)

			loss_fct = CrossEntropyLoss(reduce=False)

			start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions, span_mask)]
			start_losses = [(loss_fct(start_logits[i].unsqueeze(0), start_pairs[i][0].unsqueeze(0)) * start_pairs[i][1]) for i in range(len(start_pairs))]
			
			end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions, span_mask)]
			end_losses = [(loss_fct(end_logits[i].unsqueeze(0), end_pairs[i][0].unsqueeze(0)) * end_pairs[i][1]) for i in range(len(end_pairs))]
			
			switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch, answer_mask)]
			switch_losses = [(loss_fct(switch_logits[i].unsqueeze(0), switch_pairs[i][0].unsqueeze(0))) for i in range(len(switch_pairs))]


			before_state_start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(before_state_start_positions, before_state_span_mask)]
			before_state_start_losses = [(loss_fct(before_state_start_logits[i].unsqueeze(0), before_state_start_pairs[i][0].unsqueeze(0)) * before_state_start_pairs[i][1]) for i in range(len(before_state_start_pairs))]
			
			before_state_end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(before_state_end_positions, before_state_span_mask)]
			before_state_end_losses = [(loss_fct(before_state_end_logits[i].unsqueeze(0), before_state_end_pairs[i][0].unsqueeze(0)) * before_state_end_pairs[i][1]) for i in range(len(before_state_end_pairs))]
			
			before_state_switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(before_state_known_switch, before_state_answer_mask)]
			before_state_switch_losses = [(loss_fct(before_state_switch_logits[i].unsqueeze(0), before_state_switch_pairs[i][0].unsqueeze(0))) for i in range(len(before_state_switch_pairs))]

			total_loss = torch.sum(sum(start_losses + end_losses + switch_losses + before_state_start_losses + before_state_end_losses + before_state_switch_losses))
			return total_loss
		else:
			return start_logits, end_logits, switch_logits, before_state_start_logits, before_state_end_logits, before_state_switch_logits




class BertForQuestionAnsweringWithClassiferANDFutureContext(PreTrainedBertModel):

	def __init__(self, config, n_class=3, sequence_length=384):
		super(BertForQuestionAnsweringWithClassiferANDFutureContext, self).__init__(config)
		print(n_class)
		self.bert = BertModel(config)
		self.qa_outputs = nn.Linear(config.hidden_size, 2)
		self.qa_future_outputs = nn.Linear(config.hidden_size, 2)
		self.qa_classifier = nn.Linear(config.hidden_size*sequence_length, n_class)
		self.qa_future_context_switch = nn.Linear(config.hidden_size*sequence_length, 2)
		
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, answer_mask=None, known_switch=None, \
				future_context_input_ids=None, future_token_type_ids=None, future_attention_mask=None, future_start_position=None, future_end_position=None, future_answer_mask=None, future_context_flags=None):
		# import pdb; pdb.set_trace()
		sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		sequence_output_future_context, _ = self.bert(future_context_input_ids, future_token_type_ids, future_attention_mask, output_all_encoded_layers=False)
		
		logits = self.qa_outputs(sequence_output)
		future_logits = self.qa_future_outputs(sequence_output_future_context)
		
		start_logits, end_logits = logits.split(1, dim=-1)
		future_start_logits, future_end_logits = future_logits.split(1, dim=-1)
		
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)

		future_start_logits = future_start_logits.squeeze(-1)
		future_end_logits = future_end_logits.squeeze(-1)

		switch_logits = self.qa_classifier(sequence_output.view(-1, 384*768))
		future_context_switch_logits = self.qa_future_context_switch(sequence_output.view(-1, 384*768))

		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			
			if len(future_start_position.size()) > 1:
				future_start_position = future_start_position.squeeze(-1)
			if len(future_end_position.size()) > 1:
				future_end_position = future_end_position.squeeze(-1)

			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			answer_mask = answer_mask.type(torch.FloatTensor).cuda()
			future_context_flags= future_context_flags.type(torch.FloatTensor).cuda()
			# import pdb; pdb.set_trace()
			span_mask = torch.stack([(known_switch[x] == 0 and (future_context_flags[x] == 0)) for x in range(len(known_switch))]).type(torch.FloatTensor).cuda()
			future_span_mask = (future_context_flags == 1).type(torch.FloatTensor).cuda()

			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)
			loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)
			
			start_pairs = [(_start_positions, _span_mask) for (_start_positions, _span_mask) in zip(start_positions, span_mask)]
			start_losses = [(loss_fct(start_logits[i].unsqueeze(0), start_pairs[i][0].unsqueeze(0)) * start_pairs[i][1]) for i in range(len(start_pairs))]

			future_start_pairs = [(_future_start_positions, _future_span_mask) for (_future_start_positions, _future_span_mask) in zip(future_start_position, future_span_mask)]
			future_start_losses = [(loss_fct(future_start_logits[i].unsqueeze(0), future_start_pairs[i][0].unsqueeze(0)) * future_start_pairs[i][1]) for i in range(len(future_start_pairs))]
			
			end_pairs = [(_end_positions, _span_mask) for (_end_positions, _span_mask) in zip(end_positions, span_mask)]
			end_losses = [(loss_fct(end_logits[i].unsqueeze(0), end_pairs[i][0].unsqueeze(0)) * end_pairs[i][1]) for i in range(len(end_pairs))]

			future_end_pairs = [(_future_end_position, _future_span_mask) for (_future_end_position, _future_span_mask) in zip(future_end_position, future_span_mask)]
			future_end_losses = [(loss_fct(future_end_logits[i].unsqueeze(0), future_end_pairs[i][0].unsqueeze(0)) * future_end_pairs[i][1]) for i in range(len(future_end_pairs))]
			
			switch_pairs = [(_switch, _answer_mask) for (_switch, _answer_mask) in zip(known_switch, answer_mask)]
			switch_losses = [(loss_fct(switch_logits[i].unsqueeze(0), switch_pairs[i][0].unsqueeze(0))) for i in range(len(switch_pairs))]

			future_switch_pairs = [(_future_switch, _future_answer_mask) for (_future_switch, _future_answer_mask) in zip(future_context_switch, future_answer_mask)]
			future_switch_losses = [(loss_fct(future_context_switch_logits[i].unsqueeze(0), future_switch_pairs[i][0].unsqueeze(0))) for i in range(len(future_switch_pairs))]
			
			total_loss = torch.sum(sum(start_losses + end_losses + switch_losses + future_start_losses + future_end_losses + future_switch_losses))
			return total_loss
		else:
			return start_logits, end_logits, switch_logits, future_start_logits, future_end_logits, future_context_switch_logits


















