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


def load_vocabulary(filepath, n=2):
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
    return (p_in, p_out)


def user_bigram(users, V, prior=None, alpha=1e-2):
    perplexity = []
    print 'Estimating bigram models for each user...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        bigram = lm.bigramModel(X_train, V, prior=prior, alpha=alpha)
        perplexity.append(bigram.perplexity(X_test))
    print 'Median Perplexity = {}'.format(np.median(perplexity))
    return perplexity


def user_bigram_pp(users, V, prior, alpha=1e-2):
    pp_in, pp_out = [], []
    pred_in, pred_out = [], []
    print 'Estimating bigram models for each user...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        bigram = lm.bigramModel(X_train, V, prior=prior, alpha=alpha)
        ppIn, ppOut = bigram.perplexity_OD(X_test)
        predIn, predOut = bigram.prediction(X_test)
        pp_in.append(ppIn)
        pp_out.append(ppOut)
        pred_in.append(predIn)
        pred_out.append(predOut)
    print 'Median In Perplexity = {}'.format(np.median(pp_in))
    print 'Median Out Perplexity = {}'.format(np.median(pp_out))
    print 'Median In Prediction Accuracy = {}'.format(np.median(pred_in))
    print 'Median Out Prediction Accuracy = {}'.format(np.median(pred_out))
    return pp_in, pp_out, pred_in, pred_out


def user_trigram(users, V, prior, alpha_bi=1e-2, alpha_tri=0.1):
    pp = []
    count = 0
    print 'Estimating trigram models for each user...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        trigram = lm.trigramModel(X_train, V, prior=prior,
                                  alpha_bi=alpha_bi, alpha_tri=alpha_tri)
        perplexity = trigram.perplexity(X_test)
        pp.append(perplexity)
        count += 1
        if count % 100 == 0:
            print count
    print 'Median Perplexity = {}'.format(np.median(pp))
    return pp


def user_trigram_pp(users, V, prior, alpha_bi=1e-2, alpha_tri=0.01):
    pp_in, pp_out = [], []
    pred_in, pred_out = [], []
    count = 0
    print 'Estimating trigram models for each user...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        trigram = lm.trigramModel(X_train, V, prior=prior,
                                  alpha_bi=alpha_bi, alpha_tri=alpha_tri)
        ppIn, ppOut = trigram.perplexity_OD(X_test)
        predIn, predOut = trigram.prediction(X_test)
        pp_in.append(ppIn)
        pp_out.append(ppOut)
        pred_in.append(predIn)
        pred_out.append(predOut)
        count += 1
        if count % 100 == 0:
            print count
    print 'Median In Perplexity = {}'.format(np.median(pp_in))
    print 'Median Out Perplexity = {}'.format(np.median(pp_out))
    print 'Median In Prediction Accuracy = {}'.format(np.median(pred_in))
    print 'Median Out Prediction Accuracy = {}'.format(np.median(pred_out))
    return pp_in, pp_out, pred_in, pred_out


def user_fourgram_pp(users, V, prior, alpha_bi=1e-2, alpha_tri=0.1,
                     alpha_four=0.01):
    pp_in, pp_out = [], []
    pred_in, pred_out = [], []
    count = 0
    print 'Estimating fourgram models for each user...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        fourgram = lm.fourgramModel(X_train, V, prior=prior,
                                    alpha_bi=alpha_bi, alpha_tri=alpha_tri,
                                    alpha_four=alpha_four)
        ppIn, ppOut = fourgram.perplexity_OD(X_test)
        predIn, predOut = fourgram.prediction(X_test)
        pp_in.append(ppIn)
        pp_out.append(ppOut)
        pred_in.append(predIn)
        pred_out.append(predOut)
        count += 1
        if count % 100 == 0:
            print count
    print 'Median In Perplexity = {}'.format(np.median(pp_in))
    print 'Median Out Perplexity = {}'.format(np.median(pp_out))
    print 'Median In Prediction Accuracy = {}'.format(np.median(pred_in))
    print 'Median Out Prediction Accuracy = {}'.format(np.median(pred_out))
    return pp_in, pp_out, pred_in, pred_out


def popu_bigram(users, V, alpha):
    counter = 0
    trainSet = []
    testList = []
    print 'Estimating bigram model for the whole population...'
    for user in users:
        if (user.getActiveDays() < 60):
            continue
        X_train, X_test = build_corpus(user.tripList, 30)
        trainSet.extend(X_train)
        testList.append(X_test)
        counter += 1
    print 'Number of users = {}'.format(counter)
    print 'Number of user days in training set = {}'.format(len(trainSet))
    bigram = lm.bigramModel(trainSet, V, alpha=alpha, lowthreshold=0)
    perplexity = []
    for X_test in testList:
        perplexity.append(bigram.perplexity(X_test))
    print 'Median Perplexity = {}'.format(np.median(perplexity))
    return perplexity


def plot4(pp_list):
    names = ['In Perplexity', 'Out Perplexity',
             'In Prediction', 'Out Prediction']
    f, ax = plt.subplots(2, 2)
    for i in xrange(4):
        row = int(i/2)
        col = i % 2
        if row == 0:
            ax[row, col].set_ylim([0, 150])
            ax[row, col].hist(pp_list[i], bins=range(30))
        else:
            ax[row, col].set_ylim([0, 100])
            ax[row, col].hist(pp_list[i], bins=[j/20.0 for j in range(21)])
        median = ' (median = {0:.2f})'.format(np.median(pp_list[i]))
        ax[row, col].set_title(names[i] + median)
    plt.show()


def plot6(pp_list):
    names = ['In', 'Out', 'First In', 'First Out', 'Stop', 'Overall']
    f, ax = plt.subplots(2, 3)
    for i in xrange(len(names)):
        row = i % 2
        col = int(i/2)
        pp = pp_list[i]
        ax[row, col].hist(pp, bins=range(30))
        ax[row, col].set_title(names[i])
    plt.show()


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_dir)
    V = load_vocabulary("../Data/all_stations.csv", n=4)
    users = ur.readPanelData("../Data/sampleData_2013.csv")
    prior = construct_priors(users, V)
    pp = user_fourgram_pp(users, V, prior)
    plot4(pp)
