from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import preprocessing
import json
import time
import os
import tensorflow as tf

import transformer

flags = tf.flags

FLAGS = flags.FLAGS


def main(_) :

	batch_size = 32

	with open(os.path.join(FLAGS.wmt_folder, "vocab.en.json"), 'r') as f:
		vocab_en = json.load(f)

	with open(os.path.join(FLAGS.wmt_folder, "train.en.json"), 'r') as f:
		raw_inputs_en = json.load(f)

	with open(os.path.join(FLAGS.wmt_folder, "vocab.de.json"), 'r') as f:
		vocab_fr = json.load(f)

	with open(os.path.join(FLAGS.wmt_folder, "train.de.json"), 'r') as f:
		raw_inputs_fr = json.load(f)

	loader = transformer.input_fn_builder(raw_inputs_en, raw_inputs_fr, vocab_en, vocab_fr, batch_size)

	print("\n\n\n")

	i=0
	for data, _ in loader() :
		print(data['inputs_src'][3])
		print(data['input_masks_src'][3])
		print(data['inputs_trg'][3])
		print(data['input_masks_trg'][3])
		i+=1
		if i == 100:
			break

if __name__ == '__main__':
	tf.enable_eager_execution()
	tf.app.run()