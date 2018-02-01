# coding:utf-8
import re, os, commands, argparse, random, sys
import common

random.seed(0)

@common.timewatch()
def main(args):
  sentences = [l.replace('\n', '') for l in open(args.input_file)]
  n_all_sentences = len(sentences)

  if args.max_len:
    idxs = [i for i,s in enumerate(sentences) if not len(s.split(' ')) > args.max_len]

  else:
    idxs = [i for i in xrange(len(sentences))]

  random.shuffle(idxs)
  n_sample = min(args.n_sample, len(idxs))

  with open(args.input_file + ".m%d" % (args.max_len), 'w') as f:
    sys.stdout = f
    for idx in idxs[:n_sample]:
      print sentences[idx]
    sys.stdout = sys.__stdout__


  ## e.g. input: all.normalized.strict, origin: all.strict.origin
  original_path = re.search('(.+)\.normalized', args.input_file).group(1) + '.strict.origin'
  original_sentences = [l.replace('\n', '') for l in open(original_path)]
  if os.path.exists(original_path):
    with open(original_path + ".m%d" % (args.max_len), 'w') as f:
      sys.stdout = f
      for idx in idxs[:n_sample]:
        print original_sentences[idx]
      sys.stdout = sys.__stdout__

  sys.stderr.write("Pick up %d pieces from %d sentences (%d of them are less than or equal %d words in length).\n" % (n_sample, n_all_sentences, len(idxs), args.max_len))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", 
                      default="results/candidate_sentences/all.normalized.strict",
                      type=str, help="")
  parser.add_argument("-m", "--max_len", type=int, default=30, help="")
  parser.add_argument("-n", "--n_sample", type=int, default=10000, help="")
  args  = parser.parse_args()
  main(args)



