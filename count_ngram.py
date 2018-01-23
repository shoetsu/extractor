#coding: utf-8
import argparse
from utils import common
from utils.common import NGramVectorizer, flatten
from collections import Counter
NUM = common.NUM
def main(args):
  sents = [l.replace('\n', '').split(' ') for l in open(args.input_file)]
  vectorizer = NGramVectorizer(ngram_range=(1,4), min_freq=5)
  ngrams = vectorizer.fit_transform(sents)
  
  def _get_ngram(s, ngram_range):
    stop_words = set(['.', ',', '!', '?'])
    vocab_condition = lambda x : True if NUM in x and not stop_words.intersection(set(x)) else False
    return flatten([[tuple(s[i:i+n]) for i in xrange(len(s)-n+1) if vocab_condition(s[i:i+n])] for n in xrange(ngram_range[0], ngram_range[1]+1)])
  
  ngram_range = (1,4)
  ngrams = [_get_ngram(s, ngram_range) for s in sents])
  for ng, freq in Counter(flatten(ngrams)).most_common(10000):
    print ng, freq

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str, help="")
  args = parser.parse_args()
  main(args)
  
