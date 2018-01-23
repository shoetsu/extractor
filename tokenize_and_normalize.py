#coding:utf-8
import re, argparse, os, commands, sys
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
from utils import common

#current tmpfile: rkfj8
TAGGER_DIR = common.TAGGER_DIR
NUM = common.NUM

def main(args):
  if args.tmp_path and os.path.exists(args.tmp_path):
    sents = [l for l in open(args.tmp_path)]
  else:
    sents = [l for i, l in enumerate(open(args.input_file))]
    sys.stderr.write("Tokenizing all sentences... \n")
    sents = [word_tokenize(l) for l in sents]
    tmp_filepath = common.restore_to_tmpfile([" ".join(l) for l in sents], 
                                            tmp_dir='/tmp/extractor_tmp')
    sys.stderr.write("Tokenized sentences are restored in \'%s\'.\n" % tmp_filepath)

  if args.tmp_path and os.path.exists(args.tmp_path + '.tagged'):
    sys.stderr.write("Reading tagged tmpfile...")
  else:
    sys.stderr.write("Runnning POS Tagger...")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stanford-postagger.sh')
    cmd = "%s %s 4g" % (script_path, tmp_filepath) 
    os.system(cmd)

  pos_tags = [[x for x in l.split('\n') if x] for l in commands.getoutput('cut -f2 %s' % (tmp_filepath + '.tagged')).split('\n\n')]
  for i, (s, p) in enumerate(zip(sents, pos_tags)):
    assert len(s) == len(p)
    sent = [ss if not pp == 'CD' else NUM for ss, pp in zip(s, p)]
    print " ".join(sent)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", default="results/all.extracted",
                      type=str, help="")
  parser.add_argument("-t", "--tmp_path", default=None,
                      type=str, help="")

  args  = parser.parse_args()
  main(args)
