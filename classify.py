from learner import Learner
import re
from math import log1p as log
import file_reader
import sys


##### LEARNING PHASE #####
# Using a class 'Learner' to train our model with our train dataset

# getting alpha from arguments
alpha = int(sys.argv[1])
learner = Learner(alpha=alpha)
learner.learn()

# Chooses a sentiment by getting prior and likelihood probabilities from our Learner class.
# returns the class that maximizes the posteriori P(word|class)P(class)
def maximum_aposteriori(words):
    # argmax [logP(class) + sum(logP(word | class))]
    marginal_likelihoods = []
    classes = ['negative', 'positive']
    pos_probs = []
    neg_probs = []
    for c in classes:
        sum = log(learner.get_prior_probability(c))
        for word in words:
            prob = log(learner.get_likelihood_probability(word, c))
            if c == classes[0]:
                neg_probs.append((word, prob))
            else:
                pos_probs.append((word, prob))
            sum += prob
        marginal_likelihoods.append(sum)
    return classes[marginal_likelihoods.index(max(marginal_likelihoods))]


##### TESTING PHASE #####
# Reading articles one by one, applies maximum aposteriori to choose a sentiment.
# Calculates tables for both classes to calculate precision and recall later.
pos_tp, pos_fp, pos_fn, pos_tn = 0, 0, 0, 0
neg_tp, neg_fp, neg_fn, neg_tn = 0, 0, 0, 0
texts = file_reader.read_files('test', 'pos')
for text in texts:
    words = re.findall(r'[a-z]+\b', text)
    result = maximum_aposteriori(words)
    # print(result)
    if result == 'positive':
        pos_tp += 1
        neg_tn += 1
    else:
        pos_fn += 1
        neg_fp += 1

texts = file_reader.read_files('test', 'neg')
for text in texts:
    words = re.findall(r'\w+', text)
    result = maximum_aposteriori(words)
    # print(result)
    if result == 'positive':
        pos_fp += 1
        neg_fn += 1
    else:
        pos_tn += 1
        neg_tp += 1


##### REPORTING PHASE #####
# Reports alpha value,
# precision, recall and f measure values for each classes.
# Lastly, reports micro and macroaveraged values.
print('Alpha ' + str(alpha))

pos_precision = pos_tp / (pos_tp + pos_fp)
pos_recall = pos_tp / (pos_tp + pos_fn)
pos_f_measure = 2*pos_precision*pos_recall / (pos_precision + pos_recall)
print("[class = 'positive'] precision  " + str(pos_precision))
print("[class = 'positive'] recall  " + str(pos_recall))
print("[class = 'positive'] f_measure  " + str(pos_f_measure))


neg_precision = neg_tp / (neg_tp + neg_fp)
neg_recall = neg_tp / (neg_tp + neg_fn)
neg_f_measure = 2*neg_precision*neg_recall / (neg_precision + neg_recall)
print("[class = 'negative'] precision  " + str(neg_precision))
print("[class = 'negative'] recall  " + str(neg_recall))
print("[class = 'negative'] f_measure  " + str(neg_f_measure))

macroaveraged_precision = (pos_precision + neg_precision) / 2
print('macroaveraged precision ' + str(macroaveraged_precision))
microaveraged_precision = (pos_tp + neg_tp) / (pos_tp + neg_tp + pos_fp + neg_fp)
print('microaveraged precision ' + str(microaveraged_precision))