import os
import sys
import cPickle
import datetime
import time
import random
import tensorflow as tf
import numpy as np

from model import ConvVAEModel
from config import ConvVAEConfig
from log_utils import logger_fn
#from tensorflow.python.platform import tf_logging as logging
#logging.set_verbosity(logging.INFO)


def run_model(config, train_data, valid_data, test_data, word_to_index, index_to_word, log_manager):

	min_val_loss = 1e50
	min_test_loss = 1e50
	best_epoch = -1
	num_batches = len(train_data)//config.batch_size
	print(num_batches)

	cycle_t = num_batches * config.cycle_ep
	full_kl_step = cycle_t // 2

	model = ConvVAEModel(config, num_batches, word_to_index, index_to_word, log_manager)

	setting = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
	setting.gpu_options.allow_growth = True

	with tf.Session(config=setting) as sess:

		sess.run(tf.global_variables_initializer())
		if config.restore:
			try:
				model.saver.restore(sess, config.save_path)
				log_manager.info("Loading variables from '%s'." % config.save_path)
			except Exception as e:
				log_manager.info(e)
				log_manager.info("No saving session, using random initialization")
				sess.run(tf.global_variables_initializer())

		for epoch in range(config.num_epoch):

			log_manager.info("Starting epoch %d" % epoch)
			_, _, _ = model.run_epoch(sess, epoch, 'train', train_data, full_kl_step, cycle_t)
			val_loss, _, _ = model.run_epoch(sess, epoch, 'valid', valid_data)
			test_loss, _, _ = model.run_epoch(sess, epoch, 'test', test_data)

			if val_loss < min_val_loss:
				min_val_loss = val_loss
				best_epoch = epoch
				min_test_loss = test_loss
				model.saver.save(sess, config.save_path)
				log_manager.info("save model.")

			if config.save_last:
				model.saver.save(sess, config.save_path+'_last')
				log_manager.info("save last model.")

			log_manager.info("Min Val Loss %.4f, Min Test Loss %.4f, Best Epoch %d\n" % (min_val_loss, min_test_loss, best_epoch))


def main():

	

	print(" [*] Loading dataset.")

	data_path = "data/yelp_short_s10.p"
	data = cPickle.load(open(data_path, "rb"))

	train_data, valid_data, test_data = data[0],data[1],data[2]
	word_to_index, index_to_word = data[6], data[7]

	print(" [*] train size: %d" % len(train_data))
	print(" [*] valid size: %d" % len(valid_data))
	print(" [*] test size: %d" % len(test_data))
	print(" [*] vocabulary size: %d" % len(index_to_word))

	print("\n")

	print(" [*] Preparing hyperparameters.")
	config = ConvVAEConfig()
	config.vocab_size = len(index_to_word)

	log_file = os.path.join("log", config.anneal_type+".txt")
	log_manager = logger_fn(config.anneal_type + ".txt", log_file)

	seed = 11
	np.random.seed(seed)
	random.seed(seed)
	tf.set_random_seed(seed)

	log_manager.info(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
	log_manager.info(dict(config))
	
	run_model(config, train_data, valid_data, test_data, word_to_index, index_to_word, log_manager)


if __name__ == '__main__':
	main()
