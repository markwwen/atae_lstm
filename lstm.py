import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import argparse
import logging
import time
import collections
from WordLoader import WordLoader

class AttentionLstm(object):
    def __init__(self, wordlist, argv, aspect_num=0):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='lstm')
        parser.add_argument('--rseed', type=int, default=int(1000*time.time()) % 19491001)
        parser.add_argument('--dim_word', type=int, default=300)
        parser.add_argument('--dim_hidden', type=int, default=300)
        parser.add_argument('--dim_aspect', type=int, default=300)
        parser.add_argument('--grained', type=int, default=3, choices=[2, 3, 5])
        parser.add_argument('--dropout', type=int, default=1)
        parser.add_argument('--regular', type=float, default=0.001)
        parser.add_argument('--attention', type=int, default=0, choices=[0])
        parser.add_argument('--aspect', type=int, default=0, choices=[0])
        parser.add_argument('--word_vector', type=str, default='data/glove.840B.300d.txt')
        args, _ = parser.parse_known_args(argv)        

        self.name = args.name
        logging.info('Model init: %s' % self.name)
        
        self.srng = RandomStreams(seed=args.rseed)
        logging.info('RandomStream seed %d' % args.rseed)
        
        self.dim_word, self.dim_hidden = args.dim_word, args.dim_hidden
        self.dim_aspect = args.dim_aspect

        logging.info('dim: word=%s, hidden=%s, aspect%s' % (self.dim_word, self.dim_hidden, self.dim_aspect))

        self.grained, self.attention, self.aspect = args.grained, args.attention, args.aspect
        logging.info('grained: %s' % self.grained)
        logging.info('attention: %s' % self.attention)
        logging.info('aspect: %s' % self.aspect)

        self.dropout = args.dropout
        logging.info('dropout: %s' % self.dropout)

        self.regular = args.regular
        logging.info('l2 regular: %s' % self.regular)

        self.num = len(wordlist) + 1
        logging.info('vocabulary size: %s' % self.num)

        self.aspect_num = aspect_num
        logging.info('aspect number: %s' % self.aspect_num)

        self.init_param()
        self.load_word_vector(args.word_vector, wordlist)
        # self.load_word_information([args.negative_word, args.sentiment_word], wordlist)
        self.init_function()
    
    def init_param(self):
        def shared_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX) + b
            f = theano.function([], matrix)
            return theano.shared(f(), name=name)

        u = lambda x : 1 / np.sqrt(x)

        dimc, dimh, dima = self.dim_word, self.dim_hidden, self.dim_aspect
        dim_lstm_para = dimh + dimc

        self.Vw = shared_matrix((self.num, dimc), 'Vw', 0.01)
        self.Wi = shared_matrix((dimh, dim_lstm_para), 'Wi', u(dimh))
        self.Wo = shared_matrix((dimh, dim_lstm_para), 'Wo', u(dimh))
        self.Wf = shared_matrix((dimh, dim_lstm_para), 'Wf', u(dimh))
        self.Wc = shared_matrix((dimh, dim_lstm_para), 'Wc', u(dimh))
        self.bi = shared_matrix((dimh, ), 'bi', 0.)
        self.bo = shared_matrix((dimh, ), 'bo', 0.)
        self.bf = shared_matrix((dimh, ), 'bf', 0.)
        self.bc = shared_matrix((dimh, ), 'bc', 0.)
        self.Ws = shared_matrix((dimh, self.grained), 'Ws', u(dimh))
        self.bs = shared_matrix((self.grained, ), 'bs', 0.)
        self.h0, self.c0 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimh, dtype=theano.config.floatX)
        self.params = [self.Vw, self.Wi, self.Wo, self.Wf, self.Wc, self.bi, self.bo, self.bf, self.bc, self.Ws, self.bs]

    def init_function(self):
        sigmoid, tanh = T.nnet.sigmoid, T.tanh
        logging.info('init function...')

        self.seq_idx = T.lvector() 
        self.solution = T.matrix()
        self.seq_matrix = T.take(self.Vw, self.seq_idx, axis=0)

        h, c = T.zeros_like(self.bf, dtype=theano.config.floatX), T.zeros_like(self.bc, dtype=theano.config.floatX)

        def encode(x_t, h_fore, c_fore):
            v = T.concatenate([h_fore, x_t])            
            f_t = T.nnet.sigmoid(T.dot(self.Wf, v) + self.bf)
            i_t = T.nnet.sigmoid(T.dot(self.Wi, v) + self.bi)
            o_t = T.nnet.sigmoid(T.dot(self.Wo, v) + self.bo)
            c_next = f_t * c_fore + i_t * T.tanh(T.dot(self.Wc, v) + self.bc)
            h_next = o_t * T.tanh(c_next)
            return h_next, c_next

        scan_result, _ = theano.scan(fn=encode, sequences=[self.seq_matrix], outputs_info=[h, c])
        embedding = scan_result[0][-1]

        self.use_noise = theano.shared(np.asarray(0., dtype=theano.config.floatX))

        if self.dropout == 1:
            embedding_for_train = embedding * self.srng.binomial(embedding.shape, p = 0.5, n = 1, dtype=embedding.dtype)
            embedding_for_test = embedding * 0.5
        else:
            embedding_for_train = embedding
            embedding_for_test = embedding
            
        self.pred_for_train = T.nnet.softmax(T.dot(embedding_for_train, self.Ws) + self.bs)
        self.pred_for_test = T.nnet.softmax(T.dot(embedding_for_test, self.Ws) + self.bs)

        self.l2 = sum([T.sum(param**2) for param in self.params]) - T.sum(self.Vw**2)
        self.loss_sen = -T.tensordot(self.solution, T.log(self.pred_for_train), axes=2)
        self.loss_l2 = 0.5 * self.l2 * self.regular
        self.loss = self.loss_sen + self.loss_l2

        logging.info('getting grads...')
        grads = T.grad(self.loss, self.params)
        self.updates = collections.OrderedDict()
        self.grad = {}
        for param, grad in zip(self.params, grads):
            g = theano.shared(np.asarray(np.zeros_like(param.get_value()), \
                    dtype=theano.config.floatX))
            self.grad[param] = g
            self.updates[g] = g + grad

        logging.info("compiling func of train...")
        self.func_train = theano.function(
                inputs = [self.seq_idx, self.solution, theano.In(h, value=self.h0), theano.In(c, value=self.c0)],
                outputs = [self.loss, self.loss_sen, self.loss_l2],
                updates = self.updates,
                on_unused_input='warn')
        logging.info("compiling func of test...")
        self.func_test = theano.function(
                inputs = [self.seq_idx, theano.In(h, value=self.h0), theano.In(c, value=self.c0)],
                outputs = self.pred_for_test,
                on_unused_input='warn')
        self.func_encode = theano.function(
                inputs = [self.seq_idx, theano.In(h, value=self.h0), theano.In(c, value=self.c0)],
                outputs = embedding,
                on_unused_input='warn')
        # self.func_info = theano.function(
        #         inputs = [self.nodes, self.edges],
        #         outputs = sentiment_ratio,
        #         on_unused_input='warn')
    
    def load_word_vector(self, fname, wordlist):
        logging.info('loading word vectors...')
        loader = WordLoader()
        dic = loader.load_word_vector(fname, wordlist, self.dim_word)

        not_found = 0
        Vw = self.Vw.get_value()
        for word, index in wordlist.items():
            try:
                Vw[index] = dic[word]
            except:
                not_found += 1
        self.Vw.set_value(Vw)
        logging.info('word vectors: %s words not found.' % not_found)
    
    def load_word_information(self, fnamelist, wordlist):
        logging.info('loading word information...')
        loader = WordLoader();
        dic = loader.load_word_information(fnamelist, self.dim_lexicon)

        num = np.zeros(self.dim_lexicon, dtype=int)
        self.S = theano.shared(np.zeros(self.num, dtype=int))
        S = self.S.get_value()
        for word, index in wordlist.items():
            S[index] = dic.get(word, 0)
            num[S[index]] += 1
        self.S.set_value(S)
        logging.info('word information: %s' % num)

    def dump(self, epoch):
        import scipy.io
        mdict = {}
        for param in self.params:
            val = param.get_value()
            mdict[param.name] = val
        scipy.io.savemat('mat/%s.%s' % (self.name, epoch), mdict=mdict)