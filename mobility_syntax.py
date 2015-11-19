import numpy as np
import csv
import os
from matplotlib import pyplot as plt

import user_reader as ur
import Smartcard_NgramModel as lm


def build_corpus(records, test_size=30):
    X = []
    prev_daykey = None
    for t in records:
        if t[0] != prev_daykey:
            X.append([])
            prev_daykey = t[0]
        X[-1].append(t[2])
        X[-1].append(t[4])
    train, test = X[:-test_size], X[-test_size:]
    return train, test


def load_vocabulary(filepath, n=3):
    rd = csv.reader(open(filepath, 'rb'), delimiter=",")
    words = rd.next()
    vocab, indx = {}, 0
    for w in words:
        if (w not in vocab.keys()):
            vocab[w] = indx
            indx += 1
    # add end symbol <STOP>
    vocab["<STOP>"] = indx
    indx += 1
    # add start symbols ... <START-2> <START-1> to the index
    for j in range(1, n):
        vocab["<START-" + str(j) + ">"] = indx
        indx += 1
    return vocab


def construct_priors(users, V):
    counter = 0
    C = []
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        C.extend(X_train)
        counter += 1
    print 'Number of users = {}'.format(counter)
    print 'Number of user days in training set = {}'.format(len(C))
    bigram = lm.bigramModel(C, V, alpha=1e-4, lowthreshold=0)
    p_in, p_out = bigram.get_params()
    # print p_in.shape
    '''
    N = len(V.keys())
    wt_in = csv.writer(open("../Data/prior_in.csv", 'wb'))
    for i in xrange(N):
        print p_in[i, :].tolist()
        wt_in.writerows(p_in[i, :].tolist())
    wt_out = csv.writer(open("../Data/prior_out.csv", 'wb'))
    for i in xrange(N):
        wt_out.writerows(p_out[i, :].tolist())
    '''
    return p_in, p_out


def user_bigram(users, V, prior, alpha):
    perplexity = []
    #print 'Estimating language models...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        bigram = lm.bigramModel(X_train, V, prior=prior, alpha=alpha, lowthreshold=0)
        perplexity.append(bigram.perplexity(X_test))
    print 'Median Perplexity = {}'.format(np.median(perplexity))
    return perplexity


def user_hybrid(users, V, prior, Lambda):
    perplexity = []
    #print 'Estimating language models...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        bigram = lm.hybridModel(X_train, V, prior=prior, Lambda=Lambda)
        perplexity.append(bigram.perplexity(X_test))
    print 'Median Perplexity = {}'.format(np.median(perplexity))
    return perplexity


def user_bigram_pp(users, V, prior, alpha):
    pp_in = []
    pp_out = []
    pp_firIn = []
    pp_firOut = []
    pp_stop = []
    #print 'Estimating language models...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        bigram = lm.bigramModel(X_train, V, prior=prior, alpha=alpha, lowthreshold=0)
        ppIn, ppOut = bigram.perplexity_OD(X_test)
        ppFirIn, ppFirOut = bigram.perplexity_firstOD(X_test)
        ppStop = bigram.perplexity_stop(X_test)
        pp_in.append(ppIn)
        pp_out.append(ppOut)
        pp_firIn.append(ppFirIn)
        pp_firOut.append(ppFirOut)
        pp_stop.append(ppStop)
    #print 'Median Perplexity = {}'.format(np.median(perplexity))
    return pp_in, pp_out, pp_firIn, pp_firOut, pp_stop


def popu_bigram(users, V, alpha):
    counter = 0
    C = []
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        C.extend(X_train)
        counter += 1
    print 'Number of users = {}'.format(counter)
    print 'Number of user days in training set = {}'.format(len(C))
    bigram = lm.bigramModel(C, V, alpha=alpha, lowthreshold=0)
    
    perplexity = []
    #print 'Estimating language models...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        perplexity.append(bigram.perplexity(X_train))
    print 'Median Perplexity = {}'.format(np.median(perplexity))
    return perplexity


def evaluate_pp(users, V, prior):
    for a in [0.01]:
        print 'if alpha = {}'.format(a)
        #perplexity = user_bigram(users, V, prior, alpha=a)
        pp_in, pp_out, pp_firIn, pp_firOut, pp_stop = user_bigram_pp(users, V, prior, alpha=a)
    #plt.hist(perplexity, bins=range(30))
    #plt.xlabel('Perplexity')
    #plt.ylabel('Number of Users')
    #plt.show()
    #'''
    pp_set = [pp_in, pp_out, pp_firIn, pp_firOut, pp_stop]
    names = ['In', 'Out', 'First In', 'First Out', 'Stop']
    f, ax = plt.subplots(2, 3)
    for i in xrange(5):
        row = i%2
        col = int(i/2)
        pp = pp_set[i]
        ax[row, col].hist(pp, bins=range(30))
        ax[row, col].set_title(names[i])
    plt.show()


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_dir)
    V = load_vocabulary("../Data/all_stations.csv")
    users = ur.readPanelData("../Data/sampleData_2013.csv")
    '''
    for a in [0.05]:
        print 'if alpha = {}'.format(a)
        perplexity = popu_bigram(users, V, alpha=a)
    '''
    p_in, p_out = construct_priors(users, V)
    prior = (p_in, p_out)
    perplexity = user_hybrid(users, V, prior, Lambda=0.5)
    plt.hist(perplexity, bins=range(30))
    plt.xlabel('Perplexity')
    plt.ylabel('Number of Users')
    plt.show()
