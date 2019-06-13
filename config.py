import numpy as np
from math import floor

class ConvVAEConfig(object):
	def __init__(self):

		self.restore = False
		self.use_bow = True
		
		self.batch_size = 32
		self.maxlen = 17
		self.vocab_size = None
		self.embed_size = 256
		self.n_hid = 256
		self.z_dim = 256


		self.stride = [2, 2, 2]
		self.filter_shape = 5
		self.filter_size = 300

		
		self.lr = 1e-4
		self.num_epoch = 100
		self.init_h_only = True
		self.L = 100
		self.optimizer = 'Adam'
		self.clip_grad = None
		self.decay_rate = 0.99
		self.decay_epoch = 2
	
		self.permutation = 0
		self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)
		
		self.bp_truncation = None
		self.anneal_type = "cyc" # mono
		self.save_path = "save/vae_anneal_" + self.anneal_type
		self.log_path = "log/vae_anneal_" + self.anneal_type
		self.save_last = True
		
		self.anneal_epoch = 10
		self.max_anneal = 1.0
		self.cycle_ep = 10


		self.batch_norm = False
		self.dropout = False
		self.dropout_ratio = 0.5

		self.sent_len = self.maxlen + 2*(self.filter_shape-1)
		self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape)/self.stride[0]) + 1)
		self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape)/self.stride[1]) + 1)
		self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)


	def __iter__(self):
		for attr, value in self.__dict__.iteritems():
			yield attr, value