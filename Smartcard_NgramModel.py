import numpy as np


def buildIndex(corpus, n, lowthreshold=0):
    # initial index, to be modified later
    tmpindex, indx = {}, 0
    for snt in corpus:
        for w in snt:
            if (w not in tmpindex.keys()):
                tmpindex[w] = indx
                indx += 1

    if lowthreshold > 0:
        # eval word counts
        counts = np.zeros((indx,))
        for snt in corpus:
            for w in snt:
                counts[tmpindex[w]] += 1

        # map all the words with counts leq lowthreshold to index 0
        newindex = {}
        indx = 1  # 0 reserved for low occurence words
        for w in tmpindex.keys():
            if (counts[tmpindex[w]] <= lowthreshold):
                newindex[w] = 0
            else:
                newindex[w] = indx
                indx += 1
    else:
        newindex = tmpindex

    # add end symbol <STOP>
    newindex["<STOP>"] = indx
    indx += 1
    # add start symbols ... <START-2> <START-1> to the index
    for j in range(1, n):
        newindex["<START-" + str(j) + ">"] = indx
        indx += 1

    return newindex


def ngramGen(corpus, w2index, n):
    """ngram generator. n is the length of the ngram."""
    ngrams_in = []
    ngrams_out = []
    start_snt = ["<START-" + str(j) + ">" for j in range(n-1, 0, -1)]
    end_snt = ["<STOP>"]
    for snt in corpus:  # sentences
        s = start_snt + snt + end_snt
        for i in xrange(n - 1, len(s)):
            if (i-n+2) % 2 == 1:
                ngrams_in.append([w2index[w] for w in s[i - n + 1:i + 1]])
            elif (i-n+2) % 2 == 0:
                ngrams_out.append([w2index[w] for w in s[i - n + 1:i + 1]])
    return ngrams_in, ngrams_out


def ngramGen_firstOD(corpus, w2index, n):
    """ngram generator. n is the length of the ngram."""
    ngrams_in = []
    ngrams_out = []
    start_snt = ["<START-" + str(j) + ">" for j in range(n-1, 0, -1)]
    end_snt = ["<STOP>"]
    for snt in corpus:  # sentences
        s = start_snt + snt + end_snt
        for i in xrange(n - 1, n + 1):
            if (i-n+2) % 2 == 1:
                ngrams_in.append([w2index[w] for w in s[i - n + 1:i + 1]])
            elif (i-n+2) % 2 == 0:
                ngrams_out.append([w2index[w] for w in s[i - n + 1:i + 1]])
    return ngrams_in, ngrams_out


#############################
class bigramModel(object):
    def __init__(self, corpus, vocabulary=None, prior=None, alpha=1.0, lowthreshold=0):
        self._n = 2
        self._alpha = alpha
        if vocabulary is None:
            self._vocab = buildIndex(corpus, self._n, lowthreshold)
        else:
            self._vocab = vocabulary
        ngram_in, ngram_out = ngramGen(corpus, self._vocab, self._n)
        if prior is not None:
            prior_in, prior_out = prior[0], prior[1]
        else:
            prior_in, prior_out = None, None
        self._prob_in = self.probDbn(ngram_in, prior_in)
        self._prob_out = self.probDbn(ngram_out, prior_out)

    def probDbn(self, ngram, prior=None):
        nwords = len(self._vocab)
        if prior is None:
            prob = np.zeros((nwords, nwords)) + self._alpha
        else:
            prob = prior * self._alpha * nwords

        for w in ngram:
            prob[w[0], w[1]] += 1.0
        for i in xrange(nwords):
            prob[i, :] /= np.sum(prob[i, :])
        return prob

    def cross_entropy(self, corpus):
        LLB, N = 0.0, 0
        ngram_in, ngram_out = ngramGen(corpus, self._vocab, self._n)
        for w in ngram_in:
            LLB += np.log2(self._prob_in[w[0], w[1]])
            N += 1
        for w in ngram_out:
            LLB += np.log2(self._prob_out[w[0], w[1]])
            N += 1
        return -LLB/N

    def perplexity(self, corpus):
        return pow(2.0, self.cross_entropy(corpus))

    def perplexity_OD(self, corpus):
        LLB_in, N_in = 0.0, 0
        LLB_out, N_out = 0.0, 0
        ngram_in, ngram_out = ngramGen(corpus, self._vocab, self._n)
        for w in ngram_in:
            if w[1] == self._vocab['<STOP>']:
                continue
            LLB_in += np.log2(self._prob_in[w[0], w[1]])
            N_in += 1
        for w in ngram_out:
            LLB_out += np.log2(self._prob_out[w[0], w[1]])
            N_out += 1
        pp_in = pow(2.0, -LLB_in/N_in)
        pp_out = pow(2.0, -LLB_out/N_out)
        return pp_in, pp_out

    def perplexity_firstOD(self, corpus):
        LLB_in, LLB_out, N = 0.0, 0.0, 0
        ngram_in, ngram_out = ngramGen_firstOD(corpus, self._vocab, self._n)
        for w in ngram_in:
            LLB_in += np.log2(self._prob_in[w[0], w[1]])
            N += 1
        for w in ngram_out:
            LLB_out += np.log2(self._prob_out[w[0], w[1]])
            N += 1
        pp_in = pow(2.0, -LLB_in/N)
        pp_out = pow(2.0, -LLB_out/N)
        return pp_in, pp_out

    def perplexity_stop(self, corpus):
        LLB, N = 0.0, 0
        ngram_in, ngram_out = ngramGen(corpus, self._vocab, self._n)
        for w in ngram_in:
            if w[1] == self._vocab['<STOP>']:
                LLB += np.log2(self._prob_in[w[0], w[1]])
                N += 1
        return pow(2.0, -LLB/N)

    def get_params(self):
        return self._prob_in, self._prob_out


#############################
class trigramModel(object):
    def __init__(self, corpus, vocabulary=None, alpha=1.0, lowthreshold=0):
        self._n = 3
        self._alpha = alpha
        if vocabulary is None:
            self._vocab = buildIndex(corpus, self._n, lowthreshold)
        else:
            self._vocab = vocabulary
        ngram_in, ngram_out = ngramGen(corpus, self._vocab, self._n)
        self._prob_in = self.probDbn(ngram_in)
        self._prob_out = self.probDbn(ngram_out)

    def probDbn(self, ngram):
        nwords = len(self._vocab)
        prob = {}
        for w in ngram:
            if w[0] in prob.keys():
                if w[1] in prob[w[0]].keys():
                    prob[w[0]][w[1]][w[2]] += 1.0
                else:
                    prob[w[0]][w[1]] = np.zeros((nwords,)) + self._alpha
                    prob[w[0]][w[1]][w[2]] += 1.0
            else:
                prob[w[0]] = {}
                prob[w[0]][w[1]] = np.zeros((nwords,)) + self._alpha
                prob[w[0]][w[1]][w[2]] += 1.0
        for i in prob.keys():
            for j in prob[i].keys():
                prob[i][j] /= np.sum(prob[i][j])
        return prob

    def prob_in(self, w):
        prob = None
        if w[0] in self._prob_in.keys():
            if w[1] in self._prob_in[w[0]].keys():
                prob = self._prob_in[w[0]][w[1]][w[2]]
        return prob

    def prob_out(self, w):
        prob = None
        if w[0] in self._prob_in.keys():
            if w[1] in self._prob_out[w[0]].keys():
                prob = self._prob_out[w[0]][w[1]][w[2]]
        return prob


#############################
class hybridModel(object):
    def __init__(self, corpus, vocabulary=None, prior=None, alpha_prior=1e-2, alpha_tri=1e-5, Lambda=0.5, lowthreshold=0):
        self._vocab = vocabulary
        self._lambda = Lambda
        self._bigram = bigramModel(corpus, vocabulary, prior, alpha_prior, lowthreshold)
        self._trigram = trigramModel(corpus, vocabulary, alpha_tri, lowthreshold)

    def cross_entropy(self, corpus):
        LLB, N = 0.0, 0
        bigram_in, bigram_out = ngramGen(corpus, self._vocab, 2)
        trigram_in, trigram_out = ngramGen(corpus, self._vocab, 3)
        for i in xrange(len(bigram_in)):
            biProb = self._bigram._prob_in[bigram_in[i][0], bigram_in[i][1]]
            triProb = self._trigram.prob_in(trigram_in[i])
            if triProb is not None:
                prob = biProb * (1-self._lambda) + triProb * self._lambda
            else:
                prob = biProb
            LLB += np.log2(prob)
            N += 1
        for i in xrange(len(bigram_out)):
            biProb = self._bigram._prob_in[bigram_in[i][0], bigram_in[i][1]]
            triProb = self._trigram.prob_in(trigram_in[i])
            if triProb is not None:
                prob = biProb * (1-self._lambda) + triProb * self._lambda
            else:
                prob = biProb
            LLB += np.log2(prob)
            N += 1
        return -LLB/N

    def perplexity(self, corpus):
        return pow(2.0, self.cross_entropy(corpus))
