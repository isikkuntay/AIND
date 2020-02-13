# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""
# Sorry, didn't annotate much yet..

# Many models have been combined here:
# BERT + BIDAF-ish + UNET-ish = SOMETHING
# Have not trained it on TPU yet. Doubt it will meet time requirements.
# It must have many many bugs too. Posting here just to give some ideas. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from zipfile import ZipFile

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "gs://rundmc/simplified-nq-train.jsonl",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "cont_len", 350,
    "Length allowed for the long sentence in the BERT input. Cont + Ques should be around 500.")

flags.DEFINE_integer(
    "ques_len", 150,
    "Length allowed for the question sentence in the BERT input.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, target_conv3=None, target_conv6 = None, target_present = None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.target_conv3 = target_conv3
    self.target_conv6 = target_conv6
    self.target_present = target_present
    self.q_mask = q_mask
    self.c_mask = c_mask

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               target_conv3,
               target_conv6,
               target_present,
               q_mask,
               c_mask,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.target_conv3 = target_conv3
    self.target_conv6 = target_conv6
    self.target_present = target_present
    self.q_mask = q_mask
    self.c_mask = c_mask
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class KeplerProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    
    examples = []
    train_df = []
    
    with ZipFile('sampleDir.zip', 'r') as zipObj:
        zipObj.extractall()
    
    train_df = pd.read_json("simplified-nq-train.jsonl", orient = 'records', lines = True)
    print('Our dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
    gc.collect()
    
    for i_main, row in train.iterrows():
    
        document_text = row['document_text'].split()
        question_text = row['question_text']
        
        for candidate_no, long_answer_candidate in enumerate(row['long_answer_candidates']):
            
            target_conv3 = [0] * FLAGS.cont_len
            target_conv6 = [0] * FLAGS.cont_len
            target_present = [0] * FLAGS.cont_len
            
            q_mask = [1] * FLAGS.ques_len
            c_mask = [1] * FLAGS.cont_len

            long_ans_start_tok = long_answer_candidate['start_token']
            long_ans_end_tok = long_answer_candidate['end_token']
            long_cand_length = long_ans_end_tok - long_ans_start_tok
            
            if long_cand_length > FLAGS.cont_len:
                long_sentence = " ".join(document_text[long_ans_start_tok:long_ans_start_tok + FLAGS.cont_len)                                       
            else:
                long_sentence = " ".join(document_text[long_ans_start_tok:long_ans_end_tok)
                for i in range(long_cand_length+1,FLAG.cont_len):
                    c_mask[i] = 0
            
            if long_ans_start_tok == row['annotations'][0]['long_answer']['start_token'] and \
                len(row['annotations'][0]['short_answers']) > 0:

                #print("this is correct long answer")

                short_answer_start_token = row['annotations'][0]['short_answers'][0]['start_token']
                short_answer_end_token = row['annotations'][0]['short_answers'][0]['end_token']
                short_start_idx = short_answer_start_token-long_ans_start_tok
                short_end_idx = short_answer_end_token-long_ans_start_tok
                
                if short_end_idx < cont_len:
                    target_start[short_start_idx] = 1
                    target_end[short_end_idx] = 1

                    for i in range(short_start_idx,short_end_idx):
                        target_present[i] = 1
                else:
                    smth = "short answer beyond maximum len"
             
            ques_length = len(question_text.split())                                          
            if ques_length < FLAGS.ques_len:
                for i in range(ques_length+1,FLAGS.ques_len):
                    q_mask[i] = 0                                 
                                                       
            guid = "train-%d" % (i_main)
            text_a = tokenization.convert_to_unicode(long_sentence)
            text_b = tokenization.convert_to_unicode(question_text)
            target_start = tokenization.convert_to_unicode(target_start)
            target_end = tokenization.convert_to_unicode(target_end)
            target_present = tokenization.convert_to_unicode(target_present)
            q_mask = tokenization.convert_to_unicode(q_mask)
            c_len = tokenization.convert_to_unicode(c_mask)                                          
                                                       
            examples.append(InputExample(guid=guid, text_a=text_a,\
                                         text_b=text_b, target_start=target_start, target_end=target_end, 
                                         target_present=target_present, q_mask=q_mask, c_mask=c_mask))

    return examples        

  def get_labels(self):
    """See base class."""
    return ["target_conv3", "target_conv6", "target_present"]

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        target_conv3 = [0]*FLAGS.cont_len,
        target_conv6 = [0]*FLAGS.cont_len,
        target_present = [0]*FLAGS.cont_len,
        q_mask = [0]*FLAGS.ques_len
        c_mask = [0]*FLAGS.cont_len
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
                                             
  #We need exact length to later build the BIDAF
  tokens_a = tokens_a[0:FLAGS.cont_len]
                                                       
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)
    #We need exact length to later build the BIDAF
    tokens_b = tokens_b[0:FLAGS.cont_len]                                              
                    
  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  #The following 3 lines are redundant; just for convention                                                    
  label_conv3 = example.target_conv3
  label_conv6 = example.target_conv6
  label_present = example.target_present
  c_mask = example.c_mask
  q_mask = example.q_mask                                                       
                                                                                                              
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_start: (id = %d)" % (label_conv3))
    tf.logging.info("label_end: (id = %d)" % (label_conv6))
    tf.logging.info("label_present: (id = %d)" % (label_present))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      target_conv3 = label_conv3,
      target_conv6 = label_conv6,
      target_present = label_present,
      c_mask = c_mask,
      q_mask = q_mask,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["target_start_ids"] = create_int_feature([feature.target_conv3])
    features["target_end_ids"] = create_int_feature([feature.target_conv6])
    features["target_present_ids"] = create_int_feature([feature.target_present])
    features["q_mask"] = create_int_feature([feature.q_mask]) 
    features["c_mask"] = create_int_festure([feature.c_mask])                                                  
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "target_start_ids": tf.FixedLenFeature([], tf.int64),
      "target_end_ids": tf.FixedLenFeature([], tf.int64),
      "target_present_ids": tf.FixedLenFeature([], tf.int64),
      "c_mask": tf.FixedLenFeature([], tf.int64),
      "q_mask": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
                                                       
def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits. Discards padded entries with e^(-inf).
    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax
    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist                                                       
                                                    
def cnn_output_width(input_width, kernel_size, padding_amount, strides):
    return (input_width - kernel_size + 2*padding_amount) / strides + 1  
                                                       
def deconv_output_shape(input_batch_size, input_size_w, output_channel_size, padding):
    output_size_h = 1
    stride = 2
    filter_size_w = 2
    if padding == 'VALID':
        output_size_w = (input_size_w - 1)*stride + filter_size_w
    elif padding == 'SAME':
        output_size_w = (input_size_w - 1)*stride + 1
    else:
        raise ValueError("unknown padding")
    output_shape = tf.stack([input_batch_size, 
                                output_size_h, output_size_w, 
                                output_channel_size])                                               
    return output_shape
                                                       
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 target_conv3, target_conv6, target_present, q_mask, c_mask, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_sequence_output()
  
  output_layer_shape = modeling.get_shape_list(output_layer, expected_rank=3)                                                     
  
  batch_size = output_layer[0]
  seq_length = output_layer[1]
  hidden_size = output_layer[2]                                                       
                                                    
  hidden_size = output_layer.shape[-1].value                                              

  SW_weights = tf.get_variable(
      "similarity_weights", [1, 3*hidden_size],
      initializer=tf.contrib.layers.xavier_initializer())
                                                       
  c = output_layer[:,1:FLAGS.cont_len+1,:] #do not count the [CLS]
  q = output_layer[:,FLAGS.cont_len+2:-2,:] #do not count the [SEP] and [SEP]
                                                                                                            
  # Hidden size = 2h by convention                     
                                                       
  c_expand = tf.expand_dims(c,2)  #[B,N,1,2h]
  q_expand = tf.expand_dims(q,1)  #[B,1,M,2h]
  c_pointWise_q = c_expand * q_expand  #[B,N,M,2h]                                                     
                                                       
  c_input = tf.tile(c_expand, [1, 1, tf.shape(q)[1], 1]) #fill in to get same dims
  q_input = tf.tile(q_expand, [1, tf.shape(c)[1], 1, 1])
                                                     
  concat_input = tf.concat([c_input, q_input, c_pointWise_q], -1) # [B,N,M,6h]

  similarity=tf.reduce_sum(concat_input * self.S_W, axis=3)  #[B,N,M]
                                           
  # q_mask shape [B,M]
  # c_mask shape [B,N]                                                     
  similarity_mask = tf.expand_dims(q_mask, 1) # [B, 1, M]
                                                       
  similarity_mask = tf.tile(similarity_mask, [1,tf.shape(c)[1],1]) # [B, N, M]
                                                       
  _, c2q_dist = masked_softmax(similarity, similarity_mask, 2) # shape (B, N, M). take softmax over q
                                                       
  c2q = tf.matmul(c2q_dist, q) # shape (B, N, 2h)
                                                       
  S_max = tf.reduce_max(similarity, axis=2) # shape (B, N) ; reminder N = cont_len
                                                       
  _, c_dash_dist = masked_softmax(S_max, c_mask, 1) # distribution of shape (B, N)
                                                       
  c_dash_dist_expand = tf.expand_dims(c_dash_dist, 1) # shape (B, 1, N)
                                                       
  c_dash = tf.matmul(c_dash_dist_expand, c) # shape (B, 1, 2h)
                                                       
  c_c2q = c * c2q # shape (B, N, 2h)
  
  c_dash =  tf.tile(c_dash, [1,tf.shape(c)[1],1]) # [B, N, 2h]                                                    
                                                       
  c_c_dash = c * c_dash # shape (B, N, 2h)
                                                       
  output = tf.concat([c2q, c_c2q, c_c_dash], axis=2) # (B, N, 2h * 3)                                                      
                                                       
  output = tf.nn.dropout(output, 0.9)
                                                                                                            
  blended_reps = tf.concat([c, output], axis=2)  # (B, N, 8h)

  ### ADD MODELING LAYER .. but first add some more data                                                    
  
  pooled_output = model.get_pooled_output()  # Shape (B, 2h)

  pooled_exp = tf.expand_dims(pooled_output, 1) # shape (B, 1, 2h)                                                           
                                                       
  pooled_tile = tf.tile(pooled_tile, [1, FLAGS.cont_len, 1]) # shape (B, cont_len, 2h)                                                           
                                                                                                                                                                                                                  
  model_input = tf.concat([blended_reps, pooled_tile], 2) # shape (B, cont_len, 10h)
  
  # we will go two different routes. targets_conv will come from convolution layers and target_present from lstm..
  # the following is route 1:                                                     
                                                       
  fw_cell = tf.nn.rnn_cell.BasicLSTMCell(256)
  bw_cell = tf.nn.rnn_cell.BasicLSTMCell(256)
  rnn_outputs, rnn_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                 inputs=model_input, sequence_length=FLAGS.cont_len,
                                                                   dtype=tf.float64)
  
  rnn_outputs = tf.concat(rnn_outputs, 2) # Shape (B, cont_len, 256*2)
  rnn_outputs = tf.relu(rnn_outputs)
                                                       
  # Now copying from run_nq.py                                                       
  rnn_output_weights = tf.get_variable(
                "rnn_output_w", [1, 256],
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
  rnn_outout_bias = tf.get_variable(
                  "rnn_output_b", [1], initializer=tf.zeros_initializer())
                                                       
  rnn_outputs = tf.reshape(rnn_outputs, [batch_size*FLAGS.cont_len, hidden_size])  # shape [B*N, 2h]                                                    
                                
  rnn_logits = tf.matmul(rnn_outputs, rnn_output_weights, transpose_b=True) # shape [B*N, 1]
  
  rnn_logits = tf.nn.bias_add(rnn_logits, rnn_output_bias) # shape [B*N, 1]
                                                       
  rnn_logits = tf.reshape(rnn_logits, [batch_size, FLAGS.cont_len, 1]) #shape [B, N, 1]
                                                       
  rnn_logits = tf.squeeze(rnn_logits, axis=2) #shape [B, N]
  
  rnn_preds = tf.sigmoid(rnn_logits)                                                     
                                                       
  rnn_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_present, logits=rnn_logits) 
                                                       
  # Now Route 2: Convolutions 
  # Expand dims to make it a 3D for the convolution:
  conv_input = tf.expand_dims(model_input, axis=1)  # Change the shape to [B, 1, cont_len, 5*emb_size]                                                     
  
  #U-NET downladder filters                                                                                                         
  filter1 = tf.get_variable("conv1_filter", shape=[1, 3, hidden_size*5, 64]) # [h, w, in_size, out_size]
  filter2 = tf.get_variable("conv2_filter", shape=[1, 3, 64, 64])
  filter3 = tf.get_variable("conv3_filter", shape=[1, 3, 64, 128])
  filter4 = tf.get_variable("conv4_filter", shape=[1, 3, 128, 128])
  filter5 = tf.get_variable("conv5_filter", shape=[1, 3, 128, 256])
  filter6 = tf.get_variable("conv6_filter", shape=[1, 3, 256, 256])
                                                       
  #U-NET upladder filters                                                     
  up6_filter = tf.get_variable("up6_filter", shape=[1, 2, 256, 256])
  filter7 = tf.get_variable("conv3_filter", shape=[1, 3, 256, 256])
  up7_filter = tf.get_variable("up6_filter", shape=[1, 2, 384, 384])
  filter8 = tf.get_variable("conv3_filter", shape=[1, 3, 448, 448])
  filter9 = tf.get_variable("conv3_filter", shape=[1, 3, 448, 1])                                                     
                                                    
  # Output shapes based on default cont_len 350                                                     
  conv1 = tf.nn.conv2d(conv_input, filter=filter1, strides=[1, 1, 1, 1], padding="VALID") # shape [B, 1, 348, 64]
  conv1 = tf.nn.relu(conv1) 
  conv2 = tf.nn.conv2d(conv2, filter=filter2, strides=[1, 1, 1, 1], padding="VALID") # shape [B, 1, 346, 64]                                                     
  conv2 = tf.nn.relu(conv2) 
  maxp2 = tf.nn.max_pool(conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='VALID') # shape [B, 1, 178, 64]

  conv3 = tf.nn.conv2d(maxp2, filter=filter3, strides=[1, 1, 1, 1], padding="VALID") # shape [B, 1, 176, 128]
  conv3 = tf.nn.relu(conv3)
  conv4 = tf.nn.conv2d(conv4, filter=filter4, strides=[1, 1, 1, 1], padding="VALID") # shape [B, 1, 174, 128]
  conv4 = tf.nn.relu(conv4)
  maxp4 = tf.nn.max_pool(conv4, ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='VALID') # shape [B, 1, 87, 128]                  
                                                       
  conv5 = tf.nn.conv2d(maxp4, filter=filter5, strides=[1, 1, 1, 1], padding="VALID") # shape [B, 1, 85, 256]
  conv5 = tf.nn.relu(conv5)
  conv6 = tf.nn.conv2d(conv6, filter=filter4, strides=[1, 1, 1, 1], padding="VALID") # shape [B, 1, 83, 256]
  conv6 = tf.nn.relu(conv6)
  
  up6_output_shape = deconv_output_shape(conv6.shape[0], conv6.shape[2], conv6.shape[3], "VALID")                                                     
  conv6_up = tf.nn.conv2d_transpose(conv6, filters = up6_filter, output_shape = up6_output_shape, 
                        strides = [1, 1, 1, 1], padding = "VALID") # shape [B, 1, 166, 256]
  # Convolve until shape is equal to conv4 (174). Use padding = SAME to increase width. 
  padding = [[0,0],[0,0],[3,3],[0,0]]
  conv6_padded =  tf.pad(conv6,paddings,"CONSTANT")  # shape [B, 1, 172, 256]                                                
  conv7 =  tf.nn.conv2d(conv6_padded, filter=filter7, strides=[1, 1, 1, 1], padding="SAME") # shape [B, 1, 174, 256]                                                    
  conv7 =  tf.nn.relu(conv7)
                                                       
  conc_4n7 = tf.concat([conv4, conv7], -1) # [B, 1 , 174, 384]                                                     
  up7_output_shape = deconv_output_shape(conc_4n7.shape[0], conc_4n7.shape[2], conc_4n7.shape[3], "VALID")                                                     
  conv7_up = tf.nn.conv2d_transpose(conc_4n7, filters = up7_filter, output_shape = up7_output_shape, 
                        strides = [1, 1, 1, 1], padding = "VALID") # shape [B, 1, 348, 384]
  conc_7n1 = tf.concat([conv7_up, conv1], -1) # [B, 1 , 348, 448]
  conv8 = tf.nn.conv2d(conc_7n1, filter=filter8, strides=[1, 1, 1, 1], padding="SAME") # shape [B, 1, 350, 1]
  
  conv_logits = tf.squeeze(conv8, axis = 3) # shape [B, 1, 350]
  conv_logits = tf.squeeze(conv_logits, axis = 1) # shape [B, cont_len]
                                                       
  conv_preds = tf.nn.sigmoid(conv_logits) 
                                                       
  conv_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_present, logits=conv_logits)               
  
  with tf.variable_scope("loss"):                                                       
      total_loss = rnn_loss + conv_loss
                                                       
  return (total_loss, rnn_preds, conv_preds)                                                     
                                                       

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

# This is the most confusing one. Note that “labels” are not passed on by the model_fn_builder. 
# They are actually passed on inside tpu_estimator when it calls the model_fn. We don’t see how. 
# Apparently we need to treat labels as per example, not per batch (to be confirmed).                                                       
                                                       
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    target_start_ids = features["target_start_ids"]
    target_end_ids = features["target_end_ids"]
    target_present_ids = features["target_present_ids"]
    q_mask = features["q_mask"]
    c_mask = features["c_mask"]                                                   
                                                       
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, target_start_ids,
        target_end_ids, target_present_ids, q_mask, c_mask, num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:

       output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    else:
       raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))                                                 
                                                       
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_target_start_ids = []
  all_target_end_ids = []
  all_target_present_ids = []
  all_q_mask = []
  all_c_mask = []                                                       

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_target_start_ids.append(feature.target_start)
    all_target_end_ids.append(feature.target_end)
    all_target_present_ids.append(feature.target_present)
    all_q_mask.append(feature.q_mask)
    all_c_mask.append(feature.c_mask)                                                                                                             
                                                       
  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "target_start_ids":
            tf.constant(
                all_target_start_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_target_end_ids,
                shape=[num_examples,seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_target_present_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "": KeplerProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()