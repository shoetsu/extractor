#coding:utf-8

import sys, re, argparse, time, commands, os, collections, random
import common

random.seed(0)
def main(args):
  sents = [l.replace('\n', '') for l in open(args.input_file)]
  idx = [x for x in xrange(len(sents))]
  random.shuffle(idx)
  with open(args.input_file + '.rand', 'w') as f:
    sys.stdout = f
    for i in idx:
      print sents[i]
    sys.stdout = sys.__stdout__

  sys.stderr.write("Pick up %d pieces from %d unique sentences (%d of them are less than or equal %d words in length).\n" % (args.n_sample, n_all_sentences, len(sentences), args.max_len))

  # pos_tags = None
  # if os.path.exists(args.input_file + '.pos'):
  #   pos_tags = [l.replace('\n', '') for l in open(args.input_file + '.pos')]
  #   with open(args.input_file + '.rand.pos', 'w') as f:
  #     sys.stdout = f
  #     for i in idx:
  #       print pos_tags[i]
  #     sys.stdout = sys.__stdout__



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", 
                      default="results/candidate_sentences/all.normalized.strict",
                      type=str, help="")
  parser.add_argument("-m", "--max_len", type=int, default=30, help="")
  #parser.add_argument("-n", "--n_sample", type=int, default=10000, help="")
  args  = parser.parse_args()
  main(args)
