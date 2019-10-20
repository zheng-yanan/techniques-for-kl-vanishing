## Techniques for KL Vanishing Problem Revisited

## Background

The variational auto-encoder (VAE) has attracted much atttention from the NLP community, and has already achieved promising results in various NLP tasks.
However, a challenging problem that is commonly figured out in many works is the KL vanishing problem (also denoted as posterior collapse sometimes). 
It is expected that the VAE could learn a good latent distribution and generate well-formed texts conditioned on samples from it. 
However, the practical fact is that the VAE totally ignores the latent representations when decoding, and fails learning the latent distribution. 
In this way, since latent representations carry no useful information, the whole VAE degenerates into a standard language model.
Here we reviewing related works for handling KL vanishing.


## Solutions for KL Vanishing
1. **KL Annealing**: [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) (ICLR2016).  
    + Linear Annealing Schedule
    + Sigmoid Annealing Schedule
2. **Word Dropout**:   
[Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) (ICLR2016).
3. **Bag-of-word Loss**:   
  [Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders](https://arxiv.org/pdf/1703.10960.pdf) (2016).
4. **Inverse Auto-regressive Flow**:   
[Improved Variational Autoencoders with Inverse Autoregressive Flow](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow.pdf) (NIPS2016).
5. **Collaborative Variational Encoder-Decoder**:   
[Improving Variational Encoder-Decoders in Dialogue Generation](https://arxiv.org/abs/1802.02032) (AAAI2018).
6. **Cyclical Annealing**:   
[Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](https://arxiv.org/abs/1903.10145) (NAACL2019).


To Be Continued.
