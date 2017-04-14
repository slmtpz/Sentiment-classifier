from learner import Learner
import re
from math import log1p as log
import file_reader

learner = Learner(alpha=1)
learner.learn()


# returns the class that maximizes the posteriori P(word|class)P(class)
def maximum_aposteriori(words):
    # argmax [logP(class) + sum(logP(word | class))]
    marginal_likelihoods = []
    classes = ['negative', 'positive']
    for c in classes:
        sum = log(learner.get_prior_probability(c))
        for word in words:
            sum += log(learner.get_likelihood_probability(word, c))
        marginal_likelihoods.append(sum)
    return classes[marginal_likelihoods.index(max(marginal_likelihoods))]

pos_tp, pos_fp, pos_fn, pos_tn = 0, 0, 0, 0
neg_tp, neg_fp, neg_fn, neg_tn = 0, 0, 0, 0
texts = file_reader.read_files('test', 'pos')
for text in texts:
    words = re.findall(r'\w+', text)
    result = maximum_aposteriori(words)
    print(result)
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
    print(result)
    if result == 'positive':
        pos_fp += 1
        neg_fn += 1
    else:
        pos_tn += 1
        neg_tp += 1

pos_precision = pos_tp / (pos_tp + pos_fp)
pos_recall = pos_tp / (pos_tp + pos_fn)
pos_f_measure = 2*pos_precision*pos_recall / (pos_precision + pos_recall)
print('pos_precision  ' + str(pos_precision))
print('pos_recall  ' + str(pos_recall))
print('pos_f_measure  ' + str(pos_f_measure))


neg_precision = neg_tp / (neg_tp + neg_fp)
neg_recall = neg_tp / (neg_tp + neg_fn)
neg_f_measure = 2*neg_precision*neg_recall / (neg_precision + neg_recall)
print('neg_precision  ' + str(neg_precision))
print('neg_recall  ' + str(neg_recall))
print('neg_f_measure  ' + str(neg_f_measure))

