import os
import sys
import cPickle
import datetime
import time
import random
import tensorflow as tf
import numpy as np
from model import ConvVAEModel
from data_utils import get_minibatches_idx, prepare_data_for_cnn
from config import ConvVAEConfig
from tensorflow.python.platform import tf_logging as logging

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_model(config, data, word_to_index, index_to_word):

	num_batches = len(data) // config.batch_size
	model = ConvVAEModel(config, num_batches, word_to_index, index_to_word)

	setting = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
	setting.gpu_options.allow_growth = True

	with tf.Session(config=setting) as sess:

		sess.run(tf.global_variables_initializer())
		if config.restore:
			try:
				model.saver.restore(sess, config.save_path)
				print(" [*] Loading variables from '%s'." % config.save_path)
			except Exception as e:
				print(e)
				print(" [*] No saving session, using random initialization")
				sess.run(tf.global_variables_initializer())



		z_emb = np.zeros([len(data), config.z_dim], dtype='float32')
		mu_emb = np.zeros([len(data), config.z_dim], dtype='float32')
		kf = get_minibatches_idx(len(data), config.batch_size)
		t = 0
		for _, index in kf:
			sents_b = [data[i] for i in index]
			x_b = prepare_data_for_cnn(sents_b, config)
			mu, latent_z = sess.run([model.mu, model.latent_sample], feed_dict={model.inputs:x_b})

			z_emb[t * config.batch_size : (t+1)* config.batch_size] = np.squeeze(latent_z)
			mu_emb[t * config.batch_size : (t+1)* config.batch_size] = np.squeeze(mu)
			
			if (t+1) % 10 == 0:
				print('%d / %d' %(t+1, len(kf)))

			t += 1

	return z_emb, mu_emb


def main():
	# sys.stdout = open('log/log.txt', 'w')

	print(" [*] Loading dataset.")

	data_path = "data/yelp_short_s10.p"
	data = cPickle.load(open(data_path, "rb"))

	_, _, test_data = data[0],data[1],data[2]
	_, _, test_lab = data[3],data[4],data[5]
	word_to_index, index_to_word = data[6], data[7]

	seed = 123
	np.random.seed(seed)
	random.seed(seed)
	tf.set_random_seed(seed)
	
	print(" [*] test size: %d" % len(test_data))
	print(" [*] vocabulary size: %d" % len(index_to_word))

	print("\n")

	print(" [*] Preparing hyperparameters.")
	config = ConvVAEConfig()
	config.vocab_size = len(index_to_word)
	config.restore = True


	batch_num = 150
	sample_idx = np.random.choice(len(test_lab), config.batch_size*batch_num, replace=False)
	print(config.batch_size*batch_num)
	X = [test_data[ix] for ix in sample_idx]
	y = [test_lab[ix] for ix in sample_idx]
	y = np.array(y)

	print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
	z_emb, mu_emb = run_model(config, X, word_to_index, index_to_word)

	"""
	X_emb_2d = TSNE(n_components=2, init='pca').fit_transform(z_emb)
	np.savez('./figs/tsne_z_%s.npz'%config.anneal_type, X_emb_2d, y)
	blue = y == 0
	red = y == 1
	fig = plt.figure(figsize=(5,5))
	plt.scatter(X_emb_2d[red, 0], X_emb_2d[red, 1], c="r", s=25, edgecolor='none', alpha=0.5)
	plt.scatter(X_emb_2d[blue, 0], X_emb_2d[blue, 1], c="b", s=25, edgecolor='none', alpha=0.5)
	plt.savefig('./figs/tsne_z_%s.jpg'%config.anneal_type, bbox_inches='tight')
	plt.close(fig)
	"""
	X_emb_2d = TSNE(n_components=2, init='pca').fit_transform(mu_emb)
	np.savez('./figs/tsne_mu_%s.npz'%config.anneal_type, X_emb_2d, y)
	blue = y == 0
	red = y == 1
	fig = plt.figure(figsize=(5,5))
	plt.scatter(X_emb_2d[red, 0], X_emb_2d[red, 1], c="r", s=25, edgecolor='none', alpha=0.5)
	plt.scatter(X_emb_2d[blue, 0], X_emb_2d[blue, 1], c="b", s=25, edgecolor='none', alpha=0.5)
	plt.savefig('./figs/tsne_mu_%s.jpg'%config.anneal_type, bbox_inches='tight')
	plt.close(fig)


if __name__ == '__main__':
	main()