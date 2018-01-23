# coding:utf-8
import re, os, commands, argparse, random, sys
from utils import common

random.seed(0)
@common.timewatch()
def main(args):
  sentences = [l.replace('\n', '') for l in open(args.input_file)]
  sentences = list(set(sentences))
  n_all_sentences = len(sentences)
  if args.max_len:
    sentences = [s for s in sentences if not len(s.split(' ')) > args.max_len]
  random.shuffle(sentences)
  n_sample = args.n_sample if args.n_sample else len(sentences)
  for sent in sentences[:n_sample]:
    print sent
  sys.stderr.write("Pick up %d pieces from %d unique sentences (%d of them are less than or equal %d words in length).\n" % (args.n_sample, n_all_sentences, len(sentences), args.max_len))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file", type=str, help="")
  parser.add_argument("-m", "--max_len", type=int, default=30, help="")
  parser.add_argument("-n", "--n_sample", type=int, default=10000, help="")
  args  = parser.parse_args()
  main(args)
