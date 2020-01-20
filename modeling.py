from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class TransformerConfig(object) :
	def __init__(self,
				vocab_size_src,
				vocab_size_trg,
				max_seq_num_src,
				max_seq_num_trg,
				hidden_size,
				embedding_size,
				num_block_enc,
				num_block_dec,
				num_head) :

		self.vocab_size_src = vocab_size_src
		self.vocab_size_trg = vocab_size_trg
		self.max_seq_num_src = max_seq_num_src
		self.max_seq_num_trg = max_seq_num_trg
		self.hidden_size = hidden_size
		self.embedding_size = embedding_size
		self.num_block_enc = num_block_enc
		self.num_block_dec = num_block_dec
		self.num_head = num_head


class TransformerModel(object) :

	def __init__(self, batch_size, config, input_src_id, input_trg_id, mask_src, mask_trg) :
		
		with tf.variable_scope("word_embedding") :
			self.embedding_table_src = tf.get_variable(
				name="src_table",
				shape=[config.vocab_size_src, config.embedding_size],
				dtype=tf.float32)

			self.embedding_table_trg = tf.get_variable(
				name="trg_table",
				shape=[config.vocab_size_trg, config.embedding_size],
				dtype=tf.float32)

		input_src = tf.nn.embedding_lookup(self.embedding_table_src, input_src_id)
					+ self.PositionalEmb(config.max_seq_num_src, config.embedding_size)
		input_trg = tf.nn.embedding_lookup(self.embedding_table_trg, input_trg_id)
					+ self.PositionalEmb(config.max_seq_num_trg, config.embedding_size)

		attn_mask_enc = self.make_attn_mask(mask_src, mask_src, is_decode=False)
		attn_mask_dec = self.make_attn_mask(mask_trg, mask_trg, is_decode=True)
		attn_mask_enc_dec = self.make_attn_mask(mask_trg, mask_src, is_decode=False)

		# Encoder
		next_input = input_src

		with tf.variable_scope("encoder") :
			for layer_idx in range(config.num_block_enc) :

				with tf.variable_scope("layer_%d" % layer_idx) :
					context = self.multi_head_attention_layer(batch_size=batch_size,
													q_tensor=next_input,
													kv_tensor=next_input,
													d_model=config.embedding_size,
													n_head=config.num_head,
													attn_mask=attn_mask_enc)
					sub1 = self.add_and_norm_layer(next_input, context)

					sub2 = self.positionwise_FF(batch_size=batch_size,
										input_ff=sub1,
										hidden_size=config.hidden_layer,
										d_model=config.embedding_size)

					next_input = self.add_and_norm_layer(sub1, sub2)

		self.encoder_output = next_input

		# Decoder
		next_input = input_trg

		with tf.variable_scope("decoder") :
			for layer_idx in range(config.num_block_dec) :

				with tf.variable_scope("self_layer_%d" % layer_idx) :
					context = self.multi_head_attention_layer(batch_size=batch_size,
													q_tensor=next_input,
													kv_tensor=next_input,
													d_model=config.embedding_size,
													n_head=config.num_head,
													attn_mask=attn_mask_dec)
					sub1 = self.add_and_norm_layer(next_input, context)

				with tf.variable_scope("connect_layer_%d" % layer_idx) :
					# Connect Encoder_output to decoder
					context = self.multi_head_attention_layer(batch_size=batch_size,
													q_tensor=sub1,
													kv_tensor=self.encoder_output,
													d_model=config.embedding_size,
													n_head=config.num_head,
													attn_mask=attn_mask_enc_dec)

					sub2 = self.add_and_norm_layer(sub1, context)

					sub3 = self.positionwise_FF(batch_size=batch_size,
										input_ff=sub2,
										hidden_size=config.hidden_layer,
										d_model=config.embedding_size)

					next_input = self.add_and_norm_layer(sub2, sub3)

		self.decoder_output = next_input

	def get_encoder_output() :
		return self.encoder_output

	def get_decoder_output() :
		return self.decoder_output

	def get_embedding_table_src():
		return self.embedding_table_src

	def get_embedding_table_trg() :
		return self.embedding_table_trg

	def PositionalEmb(max_seq, d_model) :
	    half_d = (int(d_model + 1) / 2)

	    # [0, 2, 4, 6 ...] => 2i array
	    arr = np.multiply(2.0, np.arange(0, half_d, dtype=float))
	    
	    # 1 / 10000 ^ (2i / d_model)
	    even = np.power(10000.0, -np.divide(arr, d_model))
	    emb = np.repeat(even, 2)[0:d_model]
	    
	    # value in odd dimension would be converted to cosine
	    phase = [0.0, np.pi/2] * half_d
	    phase = phase[0:d_model]

	    positional_embedding = []
	    for pos in range(max_seq) :
	        # pos / 10000 ^ (2i / d_model) 
	        emb_pos = np.multiply(emb, pos)
	        
	        # odd dim => (pos/...) => (pos/...) + pi / 2
	        emb_pos = np.add(emb_pos, phase)
	        
	        positional_embedding.append(np.sin(emb_pos))
	    
	    # return shape of [1, max_seq, d_model]
	    return tf.expand_dim(tf.constant(positional_embedding), [0])


	def make_attn_mask(mask_q, mask_k, is_decode=False) :
		'''
			Args
				mask_q : mask of query of shape [N, L_q]
				mask_k : mask of key of shape [N, L_k]
				is_decode : if mask is used in decoding, multiply lower-triangular 1-value so attention can only flows to past
		'''
		batch_size = mask_q.shape.as_list()[0]

		# [N, L_q] => [N, L_q, 1]
		mask_q_r = tf.cast(tf.reshape(mask_q, [batch_size, -1, 1]), tf.int32)

		# [N, L_k] => [N, 1, L_k]
		mask_k_r = tf.cast(tf.reshape(mask_k, [batch_size, 1, -1]), tf.int32)

		# 'attt_mask' : [N, L_q, L_k]
		attn_mask = mask_q_r * mask_k_r

		if is_decode :
			# multiply lower-triangular matrix of 1
			attn_mask = attn_mask * tf.cast(tf.matrix_band_part(tf.ones_like(attn_mask), -1, 0), tf.int32)

		# value 0 : mask
		return attn_mask


	def multi_head_attention_layer(batch_size, q_tensor, kv_tensor, d_model, n_head, attn_mask) :
		'''
			Args
				q_tensor : used for building query vector, shape of [N, L_q, d_model]
				kv_tensor : used for building key-value vector, shape of [N, L_k, d_model]
				n_head : number of heads to multi-attention
				attn_mask  : used for making attention mask, shape of [N, L_q, L_k]
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
		attn_mask = attn_mask - 1
		# big minus value leads to 0.0 of softmax value
		attn_mask = attn_mask * 1000000
		
		# [N, L_q, L_k] => [N, 1, L_q, L_k]
		attn_mask = tf.expand_dims(attn_mask, axis=[1])
		attn_matrix += tf.cast(attn_mask, tf.float32)

		attn_matrix = tf.nn.softmax(attn_matrix)

		# [N, n_head, L_q, d_v]
		context_vector = tf.matmul(attn_matrix, value)

		# [N, L_q, n_head, d_h]
		context_vector = tf.transpose(context_vector, [0,2,1,3])

		# [N, L_q, d_model]
		context_vector = tf.reshape(context_vector, [batch_size, -1, d_model])

		return context_vector


	def positionwise_FF(batch_size, input_ff, hidden_size, d_model) :
		input_ff_r = tf.reshape(input_ff, [-1, d_model])

		hidden_layer = tf.layers.dense(input_ff, hidden_size, activation=tf.nn.relu, name="layer1")
		
		output_ff = tf.layers.dense(hidden_layer, d_model, activation=tf.nn.relu, name="layer2")

		return tf.reshape(output_ff, [batch_size, -1, d_model])


	def add_and_norm_layer(sublayer_input, sublayer_output):
		# begin_norm_axis => 어느축을 기점으로 평균과 분산을 구하고 normalize 할 것인가.
		# begin_params_axis => centering variable의 dimension을 정해줌. 보통 윗 값과 똑같이 두면 될듯.
		return tf.contrib.layers.layer_norm(
			inputs=sublayer_input+sublayer_output, begin_norm_axis=-1, begin_params_axis=-1)