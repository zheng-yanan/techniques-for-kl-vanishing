import numpy as np

def prepare_data_for_cnn(seqs_x, opt): 
    maxlen=opt.maxlen
    filter_h=opt.filter_shape
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    pad = filter_h -1
    x = []   
    for rev in seqs_x:    
        xx = []
        for i in xrange(pad):
            xx.append(0)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(0)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x   
    
    
def prepare_data_for_rnn(seqs_x, opt, is_add_GO = True):
    
    maxlen=opt.maxlen
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros(( n_samples, opt.sent_len)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
        if is_add_GO:
            x[idx, 0] = 1 # GO symbol
            x[idx, 1:lengths_x[idx]+1] = s_x
        else:
            x[idx, :lengths_x[idx]] = s_x
    return x   


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # if (minibatch_start != n):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
    

def add_noise(sents, opt):
    if opt.substitution == 's':
        sents_permutated= substitute_sent(sents, opt)
    elif opt.substitution == 'p':
        sents_permutated= permutate_sent(sents, opt)
    elif opt.substitution == 'a':
        sents_permutated= add_sent(sents, opt)   
    elif opt.substitution == 'd':
        sents_permutated= delete_sent(sents, opt) 
    elif opt.substitution == 'm':
        sents_permutated= mixed_noise_sent(sents, opt)
    elif opt.substitution == 'sc':
        sents_permutated = substitute_sent_char(sents, opt)
    else:
        sents_permutated= sents
        
    return sents_permutated


def permutate_sent(sents, opt):
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        if len(sent_temp) <= 1: 
            sents_p.append(sent_temp)
            continue
        idx_s= np.random.choice(len(sent_temp)-1, size=opt.permutation, replace=True)
        temp = sent_temp[idx_s[0]]
        for ii in range(opt.permutation-1):
            sent_temp[idx_s[ii]] = sent_temp[idx_s[ii+1]]
        sent_temp[idx_s[opt.permutation-1]] = temp
        sents_p.append(sent_temp)
    return sents_p
    
    
def substitute_sent(sents, opt):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        if len(sent_temp) <= 1: 
            sents_p.append(sent_temp)
            continue
        idx_s= np.random.choice(len(sent_temp)-1, size=opt.permutation, replace=True)   
        for ii in range(opt.permutation):
            sent_temp[idx_s[ii]] = np.random.choice(opt.n_words)
        sents_p.append(sent_temp)
    return sents_p       

def delete_sent(sents, opt):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        if len(sent_temp) <= 1: 
            sents_p.append(sent_temp)
            continue
        idx_s= np.random.choice(len(sent_temp)-1, size=opt.permutation, replace=True)   
        for ii in range(opt.permutation):
            sent_temp[idx_s[ii]] = -1
        sents_p.append([s for s in sent_temp if s!=-1])
    return sents_p 
    
def add_sent(sents, opt):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        if len(sent_temp) <= 1: 
            sents_p.append(sent_temp)
            continue
        idx_s= np.random.choice(len(sent_temp)-1, size=opt.permutation, replace=True)   
        for ii in range(opt.permutation):
            sent_temp.insert(idx_s[ii], np.random.choice(opt.n_words))
        sents_p.append(sent_temp[:opt.maxlen])
    return sents_p  


def mixed_noise_sent(sents, opt):
    sents = delete_sent(sents, opt)
    sents = add_sent(sents, opt)
    sents = substitute_sent(sents, opt)
    return sents
    
def substitute_sent_char(sents, opt):
    # substitute single word
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        if len(sent_temp) <= 1: 
            sents_p.append(sent_temp)
            continue
        permute_choice = [ic for ic in range(len(sent_temp)) if sent_temp[ic] != 1]
        idx_s= np.random.choice(permute_choice, size=int(opt.permutation * (len(permute_choice))), replace=True)

        for ii in range(len(idx_s)):
            sent_temp[idx_s[ii]] = np.random.choice(list(range(2,28)))
        sents_p.append(sent_temp)
    return sents_p