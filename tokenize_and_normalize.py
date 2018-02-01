#coding:utf-8
import re, argparse, os, commands, sys
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
from utils import common

## Memo: parsing all - about 3~4 hours.

#current tmpfile: rkfj8
TAGGER_DIR = common.TAGGER_DIR
NUM = common.NUM

# TODO: make them be lowercase before applying POS tagger to handle such cases as follows
# 30 Million -> CD NNP 
# 30 million -> CD CD 

def tokenize_and_pos_tagging(origin_sents, tmp_path=None, remove_tmp=True):
  tmp_filepath = ''
  if tmp_path and os.path.exists(tmp_path):
    tokenized_sents = [l for l in open(tmp_path)]
  else:
    sys.stderr.write("Tokenizing all sentences... \n")
    tokenized_sents = [word_tokenize(common.tokenize_heuristics(l).lower()) for l in origin_sents]
    #tokenized_sents = [word_tokenize(l.lower()) for l in origin_sents]
    tmp_filepath = common.restore_to_tmpfile([" ".join(l) for l in tokenized_sents], tmp_dir='/tmp/extractor_tmp')
    sys.stderr.write("Tokenized sentences are restored in \'%s\'.\n" % tmp_filepath)

  if tmp_path and os.path.exists(tmp_path + '.tagged'):
    sys.stderr.write("Reading tagged tmpfile...")
  else:
    sys.stderr.write("Runnning POS Tagger...")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stanford-postagger.sh')
    cmd = "%s %s 4g" % (script_path, tmp_filepath) 
    os.system(cmd)
  pos_tags = [[x for x in l.split('\n') if x] for l in commands.getoutput('cut -f2 %s' % (tmp_filepath + '.tagged')).split('\n\n')]

  if remove_tmp and tmp_filepath:
    os.system('rm %s %s.tagged' % (tmp_filepath, tmp_filepath))
    sys.stderr.write('Remove temporary files\n')
  return tokenized_sents, pos_tags

def convert_num(sent, pos):
  """
  Args: 
     - sent: a sentence. (list of str)
     - pos : pos-tags. (list of str)
  """
  symbols = set(['(', ')', '|', '{', '}', '<', '>']) # Characters that are wrongly recognized as 'CD'.
  sent = [NUM if pp == 'CD' and ss not in symbols else ss for j, (ss, pp) in enumerate(zip(sent, pos))
        if j == 0 or not (pp == 'CD' and pos[j-1] == 'CD')]
  pos = [pp for j, (ss, pp) in enumerate(zip(sent, pos)) if j == 0 or not (pp == 'CD' and pos[j-1] == 'CD')]
  return sent, pos

def main(args):
  # TODO: multi-processing (should that be done by shell?)
  origin_sents = [l for i, l in enumerate(open(args.input_file))]
  tmp_path = None if not args.tmp_path else args.tmp_path
  tokenized_sents, pos_tags = tokenize_and_pos_tagging(origin_sents, tmp_path)

  res_sents = []
  res_pos = []
  for i, (s, p) in enumerate(zip(tokenized_sents, pos_tags)):
    assert len(s) == len(p)
    sent = []
    pos = []
    # convert all numbers into __NUM__
    res_sents.append(sent)
    res_pos.append(pos)

  # Restore the result of normalizatoin and POS tagging.
  with open(args.input_file + '.normalized', 'w') as f:
    sys.stdout = f
    for l in res_sents:
      print ' '.join(l)
    sys.stdout = sys.__stdout__

  with open(args.input_file + '.normalized.pos', 'w') as f:
    sys.stdout = f
    for l in res_pos:
      print ' '.join(l)
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", default="results/candidate_sentences/all",
                      type=str, help="")
  parser.add_argument("-np", "--n_process", default=1, type=int)
  parser.add_argument("-t", "--tmp_path", default=None,
                      type=str, help="")

  args  = parser.parse_args()
  main(args)
