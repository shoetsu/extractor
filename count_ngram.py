#coding: utf-8
import argparse
from utils import common
from utils.common import NGramVectorizer, flatten
from collections import Counter
NUM = common.NUM
def main(args):
  window_width = 3
  sents = [l.replace('\n', '').split(' ') for l in open(args.input_file)]
  indices_around_num = [[(max(0, i-window_width), min(len(l), i+window_width)) for i, x in enumerate(l) if x == NUM] for l in sents]
  
  words = []
  for i, (idx, s) in enumerate(zip(indices_around_num, sents)):
    #print s, len(s)
    w = common.flatten([s[x[0]:x[1]] for x in idx])
    words.append(w)
    # for idxx in idx:
    #   print idxx, #print idxx, idx[idxx[0], idxx[1]],
    # print ''
  words = common.flatten(words)
  for x in sorted(Counter(words).items(), key=lambda x: -x[1])[:2000]:
    print x
  exit(1)
  #########################
  ### Count 'tokens' around NUM
  sents = common.flatten(sents)
  for x in sorted(Counter(sents).items(), key=lambda x: -x[1])[:10000]:
    print x
  exit(1)
  ########################3333

  vectorizer = NGramVectorizer(ngram_range=(1,4), min_freq=5)
  ngrams = vectorizer.fit_transform(sents)
  
  def _get_ngram(s, ngram_range):
    stop_words = set(['.', ',', '!', '?'])
    vocab_condition = lambda x : True if NUM in x and not stop_words.intersection(set(x)) else False
    return flatten([[tuple(s[i:i+n]) for i in xrange(len(s)-n+1) if vocab_condition(s[i:i+n])] for n in xrange(ngram_range[0], ngram_range[1]+1)])
  
  ngram_range = (1,4)
  ngrams = [_get_ngram(s, ngram_range) for s in sents]
  for ng, freq in Counter(flatten(ngrams)).most_common(10000):
    print ng, freq

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str, help="")
  args = parser.parse_args()
  main(args)
  
