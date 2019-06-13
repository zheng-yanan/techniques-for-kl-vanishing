import numpy as np
import tensorflow as tf

from model_utils import embedding
from model_utils import conv_encoder_3_layer
from model_utils import lstm_decoder_embedding, gru_decoder_embedding
from data_utils import get_minibatches_idx, prepare_data_for_cnn, prepare_data_for_rnn, add_noise
from tensorflow.python.platform import tf_logging as logging

class ConvVAEModel(object):

	def __init__(self, config, num_batches, word_to_index, index_to_word, log_manager):

		self.log_manager = log_manager
		self.config = config

		self.batch_size = config.batch_size
		self.seq_len = config.sent_len
		self.latent_dim = config.z_dim
		self.word_to_index = word_to_index
		self.index_to_word = index_to_word
		self.vocab_size = config.vocab_size
		self.embedding_dim = config.embed_size
		self.max_anneal = config.max_anneal
		self.anneal_epoch = config.anneal_epoch

		self.kl_w = tf.placeholder(tf.float32, shape=(), name="kl_w")
		self.is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
		self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len], name="inputs")
		self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len], name="targets")
		
		self.embedded_inputs, self.embedding = embedding(self.inputs, self.vocab_size, self.embedding_dim)
		self.embedded_inputs = tf.expand_dims(self.embedded_inputs, 3)
		self.encoded_states = conv_encoder_3_layer(self.embedded_inputs, config, is_train=self.is_train, reuse=None)

		print(self.encoded_states)
		self.encoded_states = tf.squeeze(self.encoded_states)
		print(self.encoded_states)
		bias_init = tf.constant_initializer(0.001, dtype=tf.float32)
		self.mu = mu = tf.contrib.layers.linear(self.encoded_states, num_outputs=self.latent_dim, 
			biases_initializer=bias_init, scope='mu')
		logvar = tf.contrib.layers.linear(self.encoded_states, num_outputs=self.latent_dim, 
			biases_initializer=bias_init, scope='logvar')

		epsilon = tf.random_normal(shape=tf.shape(mu))
		self.latent_sample = mu + epsilon * tf.exp(logvar * 0.5)

		# per sent
		self.kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1))

		print(self.mu)

		print(self.latent_sample)

		# bow_loss
		label_mask = tf.to_float(tf.sign(self.targets))
		bow_fc1 = tf.contrib.layers.fully_connected(self.latent_sample, 256, activation_fn=tf.tanh, scope="bow_fc1")
		bow_logits = tf.contrib.layers.fully_connected(bow_fc1, self.vocab_size, activation_fn=None, scope="bow_project")
		# [batch_size, vocab_size]
		print(bow_logits)
		tile_bow_logits = tf.tile(tf.expand_dims(bow_logits, 1), [1, self.seq_len, 1])
		# [batch_size, 1, vocab_size]
		bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=self.targets) * label_mask
		bow_loss = tf.reduce_sum(bow_loss, reduction_indices=1)
		self.avg_bow_loss  = tf.reduce_mean(bow_loss)



		self.nll_loss, self.rec_sent, _ = lstm_decoder_embedding(
			self.latent_sample, self.targets, self.embedding, config, self.is_train)
		_, self.gen_sent, _ = lstm_decoder_embedding(
			self.latent_sample, self.targets, self.embedding, config, self.is_train, feed_previous=True, is_reuse=True)


		self.elbo = self.nll_loss + self.kl_w * self.kl_loss

		if config.use_bow == False:
			aug_loss = self.elbo
		else:
			aug_loss = self.elbo + self.avg_bow_loss

		self.word_count = tf.reduce_sum(label_mask)

		self.global_step = tf.Variable(0, trainable=False)
		self.train_op = tf.contrib.layers.optimize_loss(
			aug_loss,
			global_step=self.global_step,
			optimizer="Adam",
			clip_gradients=(lambda grad: self._clip_gradients_seperate_norm(grad, config.clip_grad)) if config.clip_grad else None,
			learning_rate_decay_fn=lambda lr, g: tf.train.exponential_decay(learning_rate=lr, global_step=g, decay_rate=config.decay_rate, decay_steps=int(num_batches*config.decay_epoch)),
			learning_rate=config.lr)

		self.saver = tf.train.Saver()


	def _clip_gradients_seperate_norm(self, grads_and_vars, clip_gradients):
	
		gradients, variables = zip(*grads_and_vars)
		clipped_gradients = [clip_ops.clip_by_norm(grad, clip_gradients) for grad in gradients]
		return list(zip(clipped_gradients, variables))


	def run_epoch(self, sess, epoch, mode, data, full_kl_step=None, cycle_t=None):

		fetches = {"elbo": self.elbo, "nll_loss": self.nll_loss, "kl_loss": self.kl_loss, "word_count": self.word_count}
		if self.config.use_bow:
			fetches["bow_loss"] = self.avg_bow_loss

		if mode == "train":
			is_train = True
			fetches["train_op"] = self.train_op
		else:
			is_train = False

		total_bow_loss = 0.0
		total_elbo = 0.0
		total_nll_loss = 0.0
		total_kl_loss = 0.0
		total_count = 0.0
		word_count = 0
		local_step = 0
		global_step = epoch * (len(data) // self.batch_size)
		batch_indexes = get_minibatches_idx(len(data), self.batch_size, shuffle=True)
		
		for _, index in batch_indexes:

			local_step += 1
			
			if mode == "train":
				global_t_cyc = global_step % cycle_t
				lr_t = 0.5 * self.config.lr *(1 + np.cos(float(global_t_cyc)/cycle_t * np.pi))

			global_step += 1

			if mode == 'train':
				if self.config.anneal_type == "mono":
					beta = self.max_anneal * np.minimum(float(global_step)/(len(data) // self.batch_size * self.anneal_epoch), 1.0)
				elif self.config.anneal_type == "cyc":
					beta = self.max_anneal * np.minimum((global_t_cyc+1.)/full_kl_step, 1.)
			else:
				beta = self.max_anneal
			
			inp = [data[i] for i in index]

			targets = prepare_data_for_rnn(inp, self.config)
			# inputs = prepare_data_for_cnn(inp, self.config)
			inputs = prepare_data_for_cnn(add_noise(inp, self.config), self.config)

			feed_dict = {self.kl_w: beta, self.inputs: inputs, self.targets: targets, self.is_train:is_train}
			outs = sess.run(fetches, feed_dict=feed_dict)

			word_count 		+= outs["word_count"]
			total_count 	+= self.batch_size
			total_elbo 		+= outs['elbo'] * self.batch_size
			total_nll_loss 	+= outs['nll_loss'] * self.batch_size
			total_kl_loss 	+= outs['kl_loss'] * self.batch_size
			if self.config.use_bow:
				total_bow_loss += outs["bow_loss"] * self.batch_size

			if local_step % 100 == 0:
				if self.config.use_bow:
					self.log_manager.info("%s step %d: elbo %.4f, nll_loss %.4f, ppl %.4f, kl_loss %.4f, bow_loss %.4f, kl_w %.4f" % 
						(mode, local_step, total_elbo/total_count, total_nll_loss/total_count, np.exp(total_nll_loss/word_count) ,total_kl_loss/total_count, total_bow_loss/total_count, beta))
				else:
					self.log_manager.info("%s step %d: elbo %.4f, nll_loss %.4f, ppl %.4f, kl_loss %.4f, kl_w %.4f" % 
						(mode, local_step, total_elbo/total_count, total_nll_loss/total_count, np.exp(total_nll_loss/word_count) ,total_kl_loss/total_count, beta))

		

		index_d = np.random.choice(len(data), self.batch_size, replace=False)
		sents_d = [data[i] for i in index_d]
		# sents_d_n = sents_d
		sents_d_n = add_noise(sents_d, self.config)
		x_d_org = prepare_data_for_rnn(sents_d, self.config)
		x_d = prepare_data_for_cnn(sents_d_n, self.config)

		fetches = {"gen_sent": self.gen_sent, "rec_sent": self.rec_sent}
		feed_dict={self.kl_w: beta, self.inputs: x_d, self.targets: x_d_org, self.is_train:is_train}
		res = sess.run(fetches, feed_dict)

		for i in range(3):
			self.log_manager.info("%s Orginal: " % mode + " ".join([self.index_to_word[ix] for ix in sents_d[i] if ix!=0 and ix!=2]))
			if mode == 'train':
				self.log_manager.info("%s Recon (feedy): "%mode + " ".join([self.index_to_word[ix] for ix in res['rec_sent'][i] if ix!=0 and ix!=2]))
			self.log_manager.info("%s Recon: "%mode + " ".join([self.index_to_word[ix] for ix in res['gen_sent'][i] if ix!=0 and ix!=2]))
		
		if self.config.use_bow:
			self.log_manager.info("%s Epoch %d: elbo %.4f, nll_loss %.4f, ppl %.4f, kl_loss %.4f, bow_loss %.4f, kl_w %.4f" % 
				(mode, epoch, total_elbo/total_count, total_nll_loss/total_count, np.exp(total_nll_loss/word_count), total_kl_loss/total_count, total_bow_loss/total_count, beta))
		else:
			self.log_manager.info("%s Epoch %d: elbo %.4f, nll_loss %.4f, ppl %.4f, kl_loss %.4f, kl_w %.4f" % 
				(mode, epoch, total_elbo/total_count, total_nll_loss/total_count, np.exp(total_nll_loss/word_count), total_kl_loss/total_count, beta))
		self.log_manager.info("\n")

		return total_elbo/total_count, total_nll_loss/total_count, total_kl_loss/total_count
