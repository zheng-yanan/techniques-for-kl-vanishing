import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.contrib.legacy_seq2seq import rnn_decoder, embedding_rnn_decoder, sequence_loss, embedding_rnn_seq2seq, embedding_tied_rnn_seq2seq

from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import nn_ops, math_ops, embedding_ops, variable_scope, array_ops
from tensorflow.python.platform import tf_logging as logging





def normalizing(x, axis):
	norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
	normalized = x / (norm)
	return normalized

def embedding(inputs, vocab_size, embedding_dim, reuse=None):
	with tf.variable_scope("embedding", reuse=reuse):
		weight_init = tf.random_uniform_initializer(-0.001, 0.001)
		embedding = tf.get_variable('embedding', [vocab_size, embedding_dim], initializer=weight_init)
	norm_embedding = normalizing(embedding, 1)
	embedded_inputs = tf.nn.embedding_lookup(norm_embedding, inputs)
	return embedded_inputs, norm_embedding


def rnn_encoder(inputs, hidden_dim, cell_type, reuse=None):
	bias_init = tf.constant_initializer(0.001, dtype=tf.float32)
	weight_init = tf.random_uniform_initializer(-0.001, 0.001)
	with tf.variable_scope("rnn_encoder", reuse=reuse):
		if cell_type == "lstm":
			cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim)
			cell_bw = tf.contrib.rnn.LSTMCell(hidden_dim)
		elif cell_type == "gru":
			cell_fw = tf.contrib.rnn.GRUCell(hidden_dim)
			cell_bw = tf.contrib.rnn.GRUCell(hidden_dim)
		else:
			raise ValueError("unrecognized cell type. [lstm or gru]")

		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs , dtype=tf.float32)
		if cell_type == "lstm":
			h_fw = states[0].h
			h_bw = states[1].h
		elif cell_type == "gru":
			h_fw = state[0]
			h_bw = state[1]
		else:
			raise ValueError("unrecognized cell type. [lstm or gru]")
		hidden = tf.concat((h_fw, h_bw), 1)
		hidden = tf.nn.l2_normalize(hidden, 1)
	return hidden


def conv_encoder_3_layer(inputs, config, is_train=True, reuse=None):
	conv_acf = tf.nn.tanh
	bias_init = tf.constant_initializer(0.001, dtype=tf.float32)
	weight_init = tf.constant_initializer(0.001, dtype=tf.float32)

	filter_size = config.filter_size
	filter_shape = config.filter_shape
	embedding_dim = config.embed_size
	stride = config.stride
	sent_len3 = config.sent_len3

	inputs = regularization(inputs, config, prefix='reg_X', reuse=reuse, is_train=is_train)
	H1 = tf.contrib.layers.conv2d(
		inputs, 
		num_outputs=filter_size, 
		kernel_size=[filter_shape, embedding_dim], 
		stride = [stride[0],1],	
		weights_initializer=weight_init, 
		biases_initializer=bias_init, 
		activation_fn=None, 
		padding='VALID', 
		scope='H1_3', 
		reuse = reuse)

	H1 = regularization(H1, config, prefix='reg_H1', reuse=reuse, is_train=is_train)
	H2 = tf.contrib.layers.conv2d(
		H1,	
		num_outputs=filter_size * 2,
		kernel_size=[filter_shape, 1], 
		stride = [stride[1],1],	
		biases_initializer=bias_init, 
		activation_fn=None, 
		padding='VALID', 
		scope='H2_3',
		reuse=reuse)

	H2 = regularization(H2, config, prefix='reg_H2', reuse=reuse, is_train=is_train)
	H3 = tf.contrib.layers.conv2d(
		H2,	
		num_outputs=filter_size * 3,
		kernel_size=[sent_len3, 1], 
		activation_fn=conv_acf , 
		padding='VALID', 
		scope='H3_3', 
		reuse=reuse)

	return H3


def regularization(inputs, config, prefix="", reuse=None, is_train=None):
	acf = tf.nn.relu
	if '_X' not in prefix:
		if config.batch_norm:
			inputs = tf.contrib.layers.batch_norm(
				inputs, 
				decay=0.9, 
				center=True, 
				scale=True, 
				is_training=is_train, 
				scope=prefix+'_bn',
				reuse = reuse)
		inputs = acf(inputs)
	inputs = inputs if (not config.dropout or is_train is None) else tf.contrib.layers.dropout(inputs, keep_prob=config.dropout_ratio, scope=prefix+'_dropout')
	return inputs



def lstm_decoder_embedding(H, y, W_emb, opt, is_train, prefix = '', add_go = False, feed_previous=False, is_reuse= None, is_fed_h = True, is_sampling = False, is_softargmax = False, beam_width=None, res=None):
    #y  len* batch * [0,V]   H batch * h
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    #y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
    if add_go:
        y = tf.concat([tf.ones([opt.batch_size,1],dtype=tf.int32), y],1)

    y = tf.unstack(y, axis=1)  # 1, . , .
    # make the size of hidden unit to be n_hid
    H = tf.squeeze(H)
    if opt.init_h_only:
        H0 = layers.fully_connected(H, num_outputs = opt.n_hid, biases_initializer=biasInit, activation_fn = None, scope = prefix + 'lstm_decoder', reuse = is_reuse)
        H1 = (tf.zeros_like(H0), H0)  # initialize H and C #
    else:
        H0 = layers.fully_connected(H, num_outputs = 2*opt.n_hid, biases_initializer=biasInit, activation_fn = None, scope = prefix + 'lstm_decoder', reuse = is_reuse)
        H0_c, H0_h = tf.split(H0, num_or_size_splits=2, axis=1)
        H1 = (H0_c, H0_h)  # initialize H and C #

    y_input = [tf.concat([layers.dropout(tf.nn.embedding_lookup(W_emb, features), keep_prob=opt.dropout_ratio,is_training=is_train), H],1) for features in y] if is_fed_h   \
               else [layers.dropout(tf.nn.embedding_lookup(W_emb, features), keep_prob=opt.dropout_ratio,is_training=is_train) for features in y]

    with tf.variable_scope(prefix + 'lstm_decoder', reuse=True):
        cell = tf.contrib.rnn.LSTMCell(opt.n_hid)
    with tf.variable_scope(prefix + 'lstm_decoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_hid, opt.embed_size], initializer = weightInit)
        b = tf.get_variable('b', [opt.vocab_size], initializer = tf.random_uniform_initializer(-0.001, 0.001))
        W_new = tf.matmul(W, W_emb, transpose_b=True) # h* V

        out_proj = (W_new,b) if feed_previous else None
        decoder_res = rnn_decoder_custom_embedding(emb_inp = y_input, H=H, initial_state = H1, cell = cell, embedding = W_emb, opt = opt, feed_previous = feed_previous, output_projection=out_proj, num_symbols = opt.vocab_size, is_fed_h = is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling)
        outputs = decoder_res[0]

        if beam_width:
            #cell = rnn_cell.LSTMCell(cell_depth)
            #batch_size_tensor = constant_op.constant(opt.batch_size)
            initial_state = cell.zero_state(opt.batch_size* beam_width, tf.float32) #beam_search_decoder.tile_batch(H0, multiplier=beam_width)
            output_layer = layers_core.Dense(opt.vocab_size, use_bias=True, kernel_initializer = W_new, bias_initializer = b, activation=None)
            bsd = beam_search_decoder.BeamSearchDecoder(
                cell=cell,
                embedding=W_emb,
                start_tokens=array_ops.fill([opt.batch_size], dp.GO_ID), # go is 1
                end_token=dp.EOS_ID,
                initial_state=initial_state,
                beam_width=beam_width,
                output_layer=output_layer,
                length_penalty_weight=0.0)
            #pdb.set_trace()
            final_outputs, final_state, final_sequence_lengths = (
                decoder.dynamic_decode(bsd, output_time_major=False, maximum_iterations=opt.maxlen))
            beam_search_decoder_output = final_outputs.beam_search_decoder_output
            #print beam_search_decoder_output.get_shape()

    logits = [nn_ops.xw_plus_b(layers.dropout(out, keep_prob=opt.dropout_ratio, is_training=is_train), W_new, b) for out in outputs]  # hidden units to prob logits: out B*h  W: h*E  Wemb V*E

    if is_sampling:
        syn_sents = decoder_res[2]
        loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.ones_like(yy),tf.float32) for yy in syn_sents])
        #loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in syn_sents])
        #loss = sequence_loss(logits[:-1], syn_sents, [tf.concat([tf.ones([1]), tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32)],0) for yy in syn_sents[:-1]]) # use one more pad after EOS
        syn_sents = tf.stack(syn_sents,1)
    else:
        syn_sents = [math_ops.argmax(l, 1) for l in logits]
        syn_sents = tf.stack(syn_sents,1)
        loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.ones_like(yy),tf.float32) for yy in y[1:]])
        #loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in y[:-1]]) # use one more pad after EOS

    loss = loss * (len(logits) - 1.)
    #outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.vocab_size, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')

    # outputs : batch * len

    # save the res
    if res is not None:
        res['outputs'] = [tf.multiply(out, W) for out in outputs]


    return loss, syn_sents, logits


def gru_decoder_embedding(H, y, W_emb, opt, prefix = '', add_go = False, feed_previous=False, is_reuse= None, is_fed_h = True, is_sampling = False, is_softargmax = False, beam_width=None, res=None):
    #y  len* batch * [0,V]   H batch * h
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    #y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
    if add_go:
        y = tf.concat([tf.ones([opt.batch_size,1],dtype=tf.int32), y],1)

    y = tf.unstack(y, axis=1)  # 1, . , .
    # make the size of hidden unit to be n_hid
    if not opt.additive_noise_lambda:
        H = layers.fully_connected(H, num_outputs = opt.n_hid, biases_initializer=biasInit, activation_fn = None, scope = prefix + 'gru_decoder', reuse = is_reuse)
    H0 = tf.squeeze(H)
    # H1 = (H0, tf.zeros_like(H0))  # initialize H and C #
    H1 = H0

    y_input = [tf.concat([tf.nn.embedding_lookup(W_emb, features),H0],1) for features in y] if is_fed_h   \
               else [tf.nn.embedding_lookup(W_emb, features) for features in y]
    with tf.variable_scope(prefix + 'gru_decoder', reuse=True):
        cell = tf.contrib.rnn.GRUCell(opt.n_hid)
        # cell = tf.contrib.rnn.GRUCell(opt.maxlen)
    with tf.variable_scope(prefix + 'gru_decoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_hid, opt.embed_size], initializer = weightInit)
        b = tf.get_variable('b', [opt.vocab_size], initializer = tf.random_uniform_initializer(-0.001, 0.001))
        W_new = tf.matmul(W, W_emb, transpose_b=True) # h* V

        out_proj = (W_new,b) if feed_previous else None
        decoder_res = rnn_decoder_custom_embedding_gru(emb_inp = y_input, initial_state = H1, cell = cell, embedding = W_emb, opt = opt, feed_previous = feed_previous, output_projection=out_proj, num_symbols = opt.vocab_size, is_fed_h = is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling)
        outputs = decoder_res[0]

        if beam_width:
            #cell = rnn_cell.LSTMCell(cell_depth)
            #batch_size_tensor = constant_op.constant(opt.batch_size)
            initial_state = cell.zero_state(opt.batch_size* beam_width, tf.float32) #beam_search_decoder.tile_batch(H0, multiplier=beam_width)
            output_layer = layers_core.Dense(opt.vocab_size, use_bias=True, kernel_initializer = W_new, bias_initializer = b, activation=None)
            bsd = beam_search_decoder.BeamSearchDecoder(
                cell=cell,
                embedding=W_emb,
                start_tokens=array_ops.fill([opt.batch_size], dp.GO_ID), # go is 1
                end_token=dp.EOS_ID,
                initial_state=initial_state,
                beam_width=beam_width,
                output_layer=output_layer,
                length_penalty_weight=0.0)
            #pdb.set_trace()
            final_outputs, final_state, final_sequence_lengths = (
                decoder.dynamic_decode(bsd, output_time_major=False, maximum_iterations=opt.maxlen))
            beam_search_decoder_output = final_outputs.beam_search_decoder_output
            #print beam_search_decoder_output.get_shape()

    logits = [nn_ops.xw_plus_b(out, W_new, b) for out in outputs]  # hidden units to prob logits: out B*h  W: h*E  Wemb V*E
    if is_sampling:
        syn_sents = decoder_res[2]
        loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.ones_like(yy),tf.float32) for yy in syn_sents])
        #loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in syn_sents])
        #loss = sequence_loss(logits[:-1], syn_sents, [tf.concat([tf.ones([1]), tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32)],0) for yy in syn_sents[:-1]]) # use one more pad after EOS
        syn_sents = tf.stack(syn_sents,1)
    else:
        syn_sents = [math_ops.argmax(l, 1) for l in logits]
        syn_sents = tf.stack(syn_sents,1)
        loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.ones_like(yy),tf.float32) for yy in y[1:]])
        #loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in y[:-1]]) # use one more pad after EOS

    #outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.vocab_size, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')

    # outputs : batch * len

    # save the res
    if res is not None:
        res['outputs'] = [tf.multiply(out, W) for out in outputs]


    return loss, syn_sents, logits
def rnn_decoder_custom_embedding_gru(emb_inp,
                          initial_state,
                          cell,
                          embedding,
                          opt,
                          num_symbols,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None,
                          is_fed_h = True,
                          is_softargmax = False,
                          is_sampling = False
                          ):

  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    # embedding = variable_scope.get_variable("embedding",
    #                                         [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, initial_state, opt, output_projection,
        update_embedding_for_previous, is_fed_h=is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling) if feed_previous else None

    custom_decoder = rnn_decoder_with_sample if is_sampling else rnn_decoder_truncated

    return custom_decoder(emb_inp, initial_state, cell, loop_function=loop_function, truncate = opt.bp_truncation)

def rnn_decoder_custom_embedding(emb_inp,
                          H,
                          initial_state,
                          cell,
                          embedding,
                          opt,
                          num_symbols,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None,
                          is_fed_h = True,
                          is_softargmax = False,
                          is_sampling = False
                          ):

  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    # embedding = variable_scope.get_variable("embedding",
    #                                         [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, H, opt, output_projection,
        update_embedding_for_previous, is_fed_h=is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling) if feed_previous else None

    custom_decoder = rnn_decoder_with_sample if is_sampling else rnn_decoder_truncated

    return custom_decoder(emb_inp, initial_state, cell, loop_function=loop_function, truncate = opt.bp_truncation)


def _extract_argmax_and_embed(embedding,
                              h,
                              opt,
                              output_projection=None,
                              update_embedding=True,
                              is_fed_h = True,
                              is_softargmax = False,
                              is_sampling = False):

  def loop_function_with_sample(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    if is_sampling:
      prev_symbol_sample = tf.squeeze(tf.multinomial(prev*opt.L,1))  #B 1   multinomial(log odds)
      prev_symbol_sample = array_ops.stop_gradient(prev_symbol_sample) # important
      emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol_sample)
    else:
      if is_softargmax:
        prev_symbol_one_hot = tf.nn.log_softmax(prev*opt.L)  #B V
        emb_prev = tf.matmul( tf.exp(prev_symbol_one_hot), embedding) # solve : Requires start <= limit when delta > 0
      else:
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    
    emb_prev = tf.concat([emb_prev,h], 1) if is_fed_h else emb_prev
    if not update_embedding: #just update projection?
      emb_prev = array_ops.stop_gradient(emb_prev)
    return (emb_prev, prev_symbol_sample) if is_sampling else emb_prev

  # def loop_function(prev, _):
  #   if is_sampling:
  #     emb_prev, _ = loop_function_with_sample(prev, _)
  #   else:
  #     emb_prev = loop_function_with_sample(prev, _)
  #   return emb_prev

  return loop_function_with_sample #if is_sampling else loop_function


def rnn_decoder_truncated(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None,
                truncate=None):
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      if i >0 and truncate and tf.mod(i,truncate) == 0:
        #tf.stop_gradient(state)
        tf.stop_gradient(output)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def rnn_decoder_with_sample(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None,
                truncate=None):
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs, sample_sent = [], []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp, cur_token = loop_function(prev, i)
        sample_sent.append(cur_token)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      if i >0 and truncate and tf.mod(i,truncate) == 0:
        #tf.stop_gradient(state)
        tf.stop_gradient(output)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state, sample_sent