#coding: utf-8
import argparse, collections, re, commands, os, sys
from public import utils


def main(args):
  N = 1000000
  read_labels = utils.timewatch()(utils.read_labels)
  read_stc_file = utils.timewatch()(utils.read_stc_file)
  labels = read_labels(args.label_file)
  ids, texts = read_stc_file(args.text_file, limit_size=N)
  n_cluster = len(set(labels.values()))
  print len(ids), len(labels), n_cluster

  elements = [0 for _ in xrange(n_cluster)]
  length = [0.0 for _ in xrange(n_cluster)]
  for _id, text in zip(ids, texts):
    length[labels[_id]] += 0.01 * len(text)
    elements[labels[_id]] += 1
    
  for i in xrange(n_cluster):
    length[i] /= elements[i] 
    length[i] *= 100
  print length

if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('text_file', help ='')
  parser.add_argument('label_file', help ='')
  args = parser.parse_args()
  main(args)
