from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import os
import math
import numpy as np
import tensorflow as tf
import modeling

import preprocessing

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"wmt_folder", "./data",
	"The raw input folder for nmt training")

flags.DEFINE_integer(
	"src_max_length", 128,
	"max num of words in a sentence")

flags.DEFINE_integer(
	"trg_max_length", 128,
	"max num of words in a sentence")

flags.DEFINE_integer(
	"src_vocab_size", 50000,
	"size of vocabulary")

flags.DEFINE_integer(
	"trg_vocab_size", 50000,
	"size of vocabulary")

flags.DEFINE_float(
	"epsilon", 0.1,
	"hyperparameter for using label-smoothing")

flags.DEFINE_float(
	"warmup_steps", 4000,
	"steps to warmup in loss function")

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
	"n_block_enc", 6,
	"number of blocks")

flags.DEFINE_integer(
	"n_block_dec", 6,
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



def model_fn(features, labels, mode, params) : 

	config = modeling.TransformerConfig(
			vocab_size_src=FLAGS.src_vocab_size,
			vocab_size_trg=FLAGS.trg_vocab_size,
			max_seq_num_src=FLAGS.src_max_length,
			max_seq_num_trg=FLAGS.trg_max_length,
			hidden_size=FLAGS.hidden_size,
			embedding_size=FLAGS.d_model,
			num_block_enc=FLAGS.n_block_enc,
			num_block_dec=FLAGS.n_block_dec,
			num_head=FLAGS.n_head)

	model = modeling.TransformerModel(
			batch_size=params['batch_size'],
			config=config,
			input_src_id=features['enc_input_id'],
			input_trg_id=features['dec_input_id'],
			mask_src=features['enc_input_mask'],
			mask_trg=features['dec_input_mask'])

	decoder_output = model.get_decoder_output()


	with tf.variable_scope("output_layer") :
		# [N, seq_len, vocab_size]
		output_linear = tf.layers.dense(
			tf.reshape(decoder_output, [-1, config.embedding_size]),
			config.embedding_size,
			activation=tf.nn.relu,
			name="output_linear")

		embedding_table_trg = model.get_embedding_table_trg()

		output_bias = tf.get_variable(
			name="output_bias",
			shape=[config.embedding_size],
			initializer=tf.zeros_initializer())

		logits = tf.nn.bias_add(tf.matmul(output_layer, embedding_table_trg, transpose_b=True), output_bias)
		logSF = tf.nn.log_softmax(logits, axis=-1)

		# [N, seq_len, vocab_size]
		# predict_id would equal to left-shifed input id
		# epsilon should be positive
		predict_id = tf.one_hot(tf.roll(features['dec_input_id'], shift=-1, axis=-1), FLAGS.vocab_size_trg)
		smoothed = predict_id * (1.0 - FLAGS.epsilon) + FLAGS.epsilon / config.embedding_size

		KL_div = smoothed * (tf.math.log(smoothed) - logSF)

		num_predicts = tf.reduce_sum(features['predict_mask'])
		per_example_loss = tf.reduce_sum(smoothed * logSF, -1) * features['predict_mask']
		loss = tf.reduce_sum(per_example_loss) / num_predicts

	
	output_spec = None

	if mode == tf.estimator.ModeKeys.TRAIN :
		global_step = tf.train.get_global_step()
		global_step_f = tf.cast(global_step, tf.float32)

		learning_rate = tf.math.pow(tf.cast(config.embedding_size, tf.float32), -0.5)
						* tf.math.minimum(tf.math.pow(global_step_f, -0.5),
							global_step_f * tf.math.minimum(FLAGS.warmup_steps, -1.5))

		optimizer = tf.train.AdamOptimizer(
			learning_rate=learning_rate,
			beta_1=0.9,
			beta_2=0.98,
			epsilon=1e-09)

		train_op = optimizer.minimize(loss, global_step=global_step)

		output_spec = tf.estimator.EstimatorSpec(
			mode=mode,
			loss=loss,
			train_op=train_op)

	elif mode == tf.estimator.ModeKeys.PREDICT :
		predicted_labels = tf.argmax(logits, axis=-1)

		predictions = {
			"labels" : predicted_labels,
			"predict_mask" : features['predict_mask']
		}

		output_spec = tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions)

	else :
		raise ValueError("Only Train or Predict modes are supported: %s" % (mode))

	return output_spec



def input_fn_builder(batch_size, filenames, is_training=True, drop_remainder=True) :

	name_to_features = {
		'dec_input_id' : tf.FixedLenFeature([dec_max_len], tf.int64),
		'dec_input_mask' : tf.FixedLenFeature([dec_max_len], tf.int64),
		'enc_input_id' : tf.FixedLenFeature([enc_max_len], tf.int64),
		'enc_input_mask' : tf.FixedLenFeature([enc_max_len], tf.int64),
		'predict_mask' : tf.FixedLenFeature([dec_max_len], tf.int64)
	}

	def _decode_record(record, name_to_features) :
		example = tf.parse_single_example(record, name_to_features)

		for name in list(example.keys()) :
			t = example[name]
			if t.dtype == tf.int64 :
				t = tf.to_int32(t)
			example[name] = t 

		return example

	def input_fn() :

		dataset = tf.data.TFRecordDataset(filenames)

		if is_training :
			dataset=dataset.repeat().shuffle(1000) 

		d = d.apply(tf.contrib.data.map_and_batch(
		    lambda record : _decode_record(record, name_to_features),
		    batch_size=1))

		return dataset

	return input_fn



def main(_) :
	tf.logging.set_verbosity(tf.logging.INFO)

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