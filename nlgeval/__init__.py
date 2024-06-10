import warnings
import nltk
import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
os.environ["word2vec_path"] = os.path.join(current_dir, "metric/word2vec")
from metric import NLGEval

omit_metrics = [
    'CIDEr',
    'METEOR',
]

def calc_avglen(hyp_list):
    cnt = 0
    for each in hyp_list:
        cnt += len(each.split(' '))

    return cnt/len(hyp_list)

def calc_nlg_metrics(decoder_preds, decoder_labels, name=None,no_glove=False):
    ref_list = []
    hyp_list = []
    for ref, hyp in zip(decoder_labels, decoder_preds):
        ref = ' '.join(nltk.word_tokenize(ref.lower()))
        hyp = ' '.join(nltk.word_tokenize(hyp.lower()))
        if len(hyp) == 0:
            hyp = '&'
        ref_list.append(ref)
        hyp_list.append(hyp)
    metric = NLGEval(no_glove=no_glove, metrics_to_omit=omit_metrics)
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    for k in range(1,5):
        metric_res[f'Distinct-{k}'] = calc_distinct_k(hyp_list, k, name)
    metric_res['avg_len'] = calc_avglen(hyp_list)
    return metric_res

def calc_distinct_k(hyps, k, name):
    d = {}
    tot = 0
    for sen in hyps:
        tokens = nltk.word_tokenize(sen.lower())
        for i in range(0, len(tokens)-k+1):
            key = tuple(tokens[i:i+k])
            d[key] = 1
            tot += 1
    if tot > 0:
        dist = len(d) / tot
    else:
        warnings.warn('the distinct is invalid')
        warnings.warn(name)
        dist = 0.
    return dist