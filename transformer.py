from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import os
import math
import numpy as np
import tensorflow as tf

import preprocessing

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"wmt_folder", "./data",
	"The raw input folder for nmt training")

flags.DEFINE_integer(
	"src_max_length", 50,
	"max num of words in a sentence")

flags.DEFINE_integer(
	"trg_max_length", 50,
	"max num of words in a sentence")

flags.DEFINE_string(
	"output_dir", None,
	"The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
	"How often to save the model checkpoint.")

flags.DEFINE_bool(
	"do_train", True,
	"True if do train")

flags.DEFINE_float(
	"learning_rate", 1e-2,
	"learning_rate for training")

flags.DEFINE_float(
	"decay_rate", 5e-5,
	"True if do train")

flags.DEFINE_bool(
	"do_predict", True,
	"True if do predict")

flags.DEFINE_integer(
	"d_model", 512,
	"embedgging size for word")

flags.DEFINE_integer(
	"n_block", 6,
	"number of blocks")

flags.DEFINE_integer(
	"n_head", 8,
	"number of heads")

flags.DEFINE_integer(
	"hidden_size", 2048,
	"hidden size of inner layer in position-wise FF layer")

flags.DEFINE_float(
	"num_epochs", 28.0,
	"number of epochs")

flags.DEFINE_integer(
	"train_batch_size", 64,
	"number of batch_size for train")

flags.DEFINE_integer(
	"predict_batch_size", 64,
	"number of batch_size for predict")


def multi_head_attention_layer(batch_size, q_tensor, kv_tensor, d_model, n_head, attn_masks) :
	'''
		Args
			q_tensor : used for building query vector, shape of [N, L_q, d_model]
			kv_tensor : used for building key-value vector, shape of [N, L_k, d_model]
			n_head : number of heads to multi-attention
			attn_masks  : used for making attention mask, shape of [N, L_q, L_k]
	'''
	# assert that d_q = d_k = d_v = d_h

	q_tensor = tf.reshape(q_tensor, [-1, d_model])
	kv_tensor = tf.reshape(kv_tensor, [-1, d_model])

	query = tf.layers.dense(
			q_tensor,
			d_model,
			activation=None,
			name="query_weight")

	key = tf.layers.dense(
			kv_tensor,
			d_model,
			activation=None,
			name="key_weight")

	value = tf.layers.dense(
			kv_tensor,
			d_model,
			activation=None,
			name="value_weight")

	d_h = d_model // n_head

	# [-1, d_model] => [N, L, n_head, d_h] => [N, n_head, L, d_h]
	query = tf.transpose(tf.reshape(query, [batch_size, -1, n_head, d_h]), [0,2,1,3])
	key = tf.transpose(tf.reshape(key, [batch_size, -1, n_head, d_h]), [0,2,1,3])
	value = tf.transpose(tf.reshape(value, [batch_size, -1, n_head, d_h]), [0,2,1,3])

	# [N, n_head, L_q, L_k]
	attn_matrix = tf.matmul(query, key, transpose_b=True)
	attn_matrix = tf.divide(attn_matrix, 1.0 / math.sqrt(float(n_h)))

	# 1 for attending, 0 for not attending. we will focus on latter
	attn_mask = attn_mask - 1.0
	# low minus value leads to 0.0 of softmax value
	attn_mask = attn_mask * 1000000.0
	# [N, L_q, L_k] => [N, 1, L_q, L_k]
	attn_mask = tf.expand_dims(attn_mask, axis=[1])
	attn_matrix += attn_mask

	attn_matrix = tf.nn.softmax(attn_matrix)

	# [N, n_head, L, d_h]
	context_vector = tf.matmul(attn_matrix, value)

	# [N, L, n_head, d_h]
	context_vector = tf.transpose(context_vector, [0,2,1,3])

	# [N, L, d_model]
	context_vector = tf.reshape(context_vector, [batch_size, -1, d_model])

	return context_vector



def make_attn_mask(mask_q, mask_k, is_decode=False) :
	batch_size = mask_k.shape.as_list()[0]
	k_len = mask_k.shape.as_list()[1]

	# [N, L_q] => [N, L_q, L_k]
	# not to attend to <pad>
	attn_mask = tf.tile(mask_q, [1, k_len])

	if is_decode :
		# L_q = L_k
		# Mask to attend only to front words
		# => plus lower trianguler matrix
		tril = np.zeros((batch_size, k_len, k_len))
		for i in range(0, len(tril)) :
			for j in range(0, len(tril[i])):
				for k in range(0, j+1) :
					tril[i][j][k] = 1

		attn_mask = attn_mask * tf.constant(tril)

	return attn_mask




def modeling(inputs_src, inputs_trg, vocab_size_src, vocab_size_trg, input_masks_src, input_masks_trg, embedding_size, hidden_size, mode) :
	'''
		Args
			inputs_src : ids of src tokens of shape [N, L]
			inputs_trg : ids of trg tokens of shape [N, L]
			vocab_size : size of vocab
			input_masks : mask for inputs of shape[N, L]
			embedding_size : size of word embedding
			hidden_size : size of inner layer in FF
	'''

	with tf.variable_scope("word_embedding") :
		embedding_table_src = tf.get_variable(
			name="embedding_table_src",
			shape=[vocab_size_src, embedding_size],
			dtype=tf.float32)

		embedding_table_trg = tf.get_variable(
			name="embedding_table_trg",
			shape=[vocab_size_trg, embedding_size],
			dtype=tf.float32)

	src_emb = tf.nn.embedding_lookup(embedding_table_src, inputs_src)
	trg_emb = tf.nn.embedding_lookup(embedding_table_trg, inputs_trg)

	# positional embedding (TODO)




	# 'Ax' = [N, embedding_size]
	Ax = tf.matmul(input_tensor, embedding_table)

	with tf.variable_scope("hidden_layer") :
		# 'BAX' = [N, hidden_size]
		BAx = tf.layers.dense(
				Ax,
				hidden_size,
				activation=None,
				name="hidden_weight")

	with tf.variable_scope("softmax_loss") :
		# 'logits' = [N, n_class]
		logits = tf.layers.dense(
					BAx,
					n_class,
					activation=None,
					name="softmax_weight")

		# only logits needed for predict
		if mode == tf.estimator.ModeKeys.PREDICT :
			return logits, -1

		# 'labels' => [N, n_class]
		labels = tf.reshape(labels, [-1])
		labels = tf.one_hot(labels, n_class)

		logSF = tf.nn.log_softmax(logits)
		loss = -tf.reduce_mean(tf.reduce_sum(labels*logSF, axis=-1))

	return logits, loss

'''

def model_fn(features, labels, mode, params) : 

	logit, loss = modeling(input_tensor=features["input_tensor"],
				labels=labels,
				vocab_size=params['vocab_size'],
				embedding_size=params['embedding_size'],
				hidden_size=params['hidden_size'],
				n_class=params['n_class'],
				mode=mode)

	output_spec = None

	if mode == tf.estimator.ModeKeys.TRAIN :
		global_step=tf.train.get_global_step()

		# linearly decaying lr
		learning_rate = (1 - tf.cast(global_step, tf.float32) * tf.constant(FLAGS.decay_rate)) * tf.constant(FLAGS.learning_rate)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)

		output_spec = tf.estimator.EstimatorSpec(
					mode=mode,
					loss=loss,
					train_op=train_op)
		
	elif mode == tf.estimator.ModeKeys.PREDICT :
		predicted_labels = tf.argmax(logit, axis=-1)

		predictions = {
			"labels" : tf.reshape(features["labels"], [-1]),
			"labels_p" : predicted_labels
		}

		output_spec = tf.estimator.EstimatorSpec(
					mode=mode,
					predictions=predictions)

	else :
		raise ValueError("Only Train or Predict modes are supported: %s" % (mode))

	return output_spec

'''


def input_fn_builder(raw_inputs_src, raw_inputs_trg, vocab_src, vocab_trg, batch_size, is_training=True, drop_remainder=True) :

	def map2id(raw_inputs, vocab, max_length) :
		examples = []
		for i in range(0, len(raw_inputs)) :
			example = []
			for j in range(0, len(raw_inputs[i])) :
				if raw_inputs[i][j] in vocab :
					example.append(int(vocab.index(raw_inputs[i][j])))
				else :
					example.append(int(vocab.index("<unk>")))

			examples.append(example)

		return examples
	
	inputs_src = map2id(raw_inputs_src, vocab_src, FLAGS.src_max_length)
	inputs_trg = map2id(raw_inputs_trg, vocab_trg, FLAGS.trg_max_length)

	def input_masks(inputs, max_length, vocab_size) :
		input_masks = []
		for i in range(0, len(inputs)) :
			input_masks.append([1] * min(max_length, len(inputs[i])) + [0] * max(max_length - len(inputs[i]), 0))

			if len(inputs[i]) < max_length :
				# append id mapping to "<pad>"
				inputs[i] += [vocab_size - 1] * (max_length - len(inputs[i]))
			else :
				inputs[i] = inputs[i][0:max_length]

		return input_masks

	input_masks_src = input_masks(inputs_src, FLAGS.src_max_length)
	input_masks_trg = input_masks(inputs_trg, FLAGS.trg_max_length)

	features = {
		"inputs_src" : inputs_src,
		"inputs_trg" : inputs_trg,
		"input_masks_src" : input_masks_src,
		"input_masks_trg" : input_masks_trg
	}

	def input_fn() :

		dataset = tf.data.Dataset.from_tensor_slices((features, [i for i in range(0, len(inputs_src))]))

		if is_training :
			dataset=dataset.repeat().shuffle(1000) 

		dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

		return dataset

	return input_fn



def main(_) :
	tf.logging.set_verbosity(tf.logging.INFO)

	with open(os.path.join(FLAGS.wmt_folder, "vocab.en.json"), 'r') as f:
		vocab_en = json.load(f)
		vocab_en.append("<pad>")

	with open(os.path.join(FLAGS.wmt_folder, "vocab.de.json"), 'r') as f:
		vocab_de = json.load(f)
		vocab_de.append("<pad>")

	tf.gfile.MakeDirs(FLAGS.output_dir)
 
	run_config = tf.estimator.RunConfig(
			model_dir=FLAGS.output_dir,
			save_checkpoints_steps=FLAGS.save_checkpoints_steps)



	'''

	estimator = tf.estimator.Estimator(
			model_fn=model_fn,
			config=run_config,
			params={
				"vocab_size" : vocab_size,
				"embedding_size" : FLAGS.embedding_size,
				"hidden_size" : FLAGS.hidden_size,
				"n_class" : FLAGS.num_class
			})



	if FLAGS.do_train :
		with open(FLAGS.train_file, 'r') as f:
			data = json.load(f)
			data = data
			num_examples = len(data)

		batch_size = FLAGS.train_batch_size

		tf.logging.info("Loading Train data...")
		raw_data, raw_len, raw_labels = raw_data_load(data, vocab, vocab_hash)
		train_steps = int(float(num_examples) / float(batch_size) * FLAGS.num_epochs)

		tf.logging.info("\n\n***** Running training *****")
		tf.logging.info("  Num examples = %d", num_examples)
		tf.logging.info("  Batch size = %d", batch_size)
		tf.logging.info("  Num steps = %d", train_steps)

		current_time = time.time()
		estimator.train(
				input_fn=lambda: input_fn(raw_data, raw_len, raw_labels, batch_size, vocab_size, is_training=True, drop_remainder=True),
				max_steps=train_steps)

		tf.logging.info("Trainning time : %.2f minutes\n\n\n", ((time.time() - current_time) / 60.0))
		

	if FLAGS.do_predict :
		with open(FLAGS.predict_file, 'r') as f:
			data = json.load(f)
			data = data
			num_examples = len(data)

		batch_size = FLAGS.predict_batch_size

		tf.logging.info("Loading Predict data...")
		raw_data, raw_len, raw_labels = raw_data_load(data, vocab, vocab_hash)

		tf.logging.info("\n\n***** Running Predictions *****")
		tf.logging.info("  Num examples = %d", num_examples)
		tf.logging.info("  Batch size = %d", batch_size)

		T = 0
		F = 0
		for predictions in estimator.predict(
			input_fn=lambda: input_fn(raw_data, raw_len, raw_labels, batch_size, vocab_size, is_training=False, drop_remainder=False),
			yield_single_examples=False):

			for i in range(0, len(predictions['labels_p'])) :
				if (T + F) % 1000 == 0 :
					tf.logging.info("Processing example: %d" % (T+F))

				if predictions['labels_p'][i] == predictions['labels'][i] :
					T+=1
				else :
					F+=1

		tf.logging.info("Accuracy : %.3f", (T/(T+F)))


'''
if __name__ == "__main__":
	flags.mark_flag_as_required("output_dir")
	tf.app.run()