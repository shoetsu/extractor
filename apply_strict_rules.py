#coding: utf-8
import sys, re, argparse, time, commands, os, collections, random
import spacy #, sense2vec
from utils import common, currency
from nltk.stem import WordNetLemmatizer

random.seed(0)
wnl = WordNetLemmatizer()
NUM = common.NUM
c_symbols, c_names = currency.get_currency_tokens()

def contain_synonym_around_num(sentence, num_indices, window_width=4):
  '''
  sentence: List of string. (a lemmatized and tokenized sentence)
  num_indices: List of integer. (the indices where NUM is)
  '''
  # TODO: remove words that can be irrelevant.
  # synonyms = set([
  #   'amount', 'bill', 'cost', 'demand', 'discount', 'estimate', 'expenditure', 'expense', 'fare', 'fee', 'figure', 'output', 'pay', 'payment', 'premium', 'rate', 'return', 'tariff', 'valuation', 'worth', 'appraisal', 'assessment', 'barter', 'bounty', 'ceiling', 'charge', 'compensation', 'consideration', 'damage', 'disbursement', 'dues', 'exaction', 'hire', 'outlay', 'prize', 'quotation', 'ransom', 'reckoning', 'retail', 'reward', 'score', 'sticker', 'tab', 'ticket', 'toll', 'tune', 'wages', 'wholesale', 'appraisement', 
  # ])

  # synonyms = [
  #   'amount', 'bill', 'cost', 'demand', 'discount', 'expenditure', 
  #   'expense', 'fare', 'fee', 'pay', 'payment', 'premium',
  #   'tariff', 'valuation', 'worth', 'appraisal', 'assessment', 'barter', 'bounty',
  #   'ceiling', 'charge', 'compensation', 'disbursement', 'dues',
  #   'exaction', 'hire', 'outlay', 'prize', 'quotation', 'ransom', 'reckoning', 
  #   'retail', 'reward', 'toll', 'tune', 'wages', 'wholesale',
  #   'appraisement'

  # ]
  synonyms = ['price', 'toll', 'cost', 'pay', 'worth', 'sell', 'charge', 'expend']

  # Whether words at the left side of NUM contain one of synonyms. 
  # (e.g. 'cost $ 30')
  words = common.flatten([sentence[max(0, idx-window_width):idx] 
                          for idx in num_indices])
  return set(synonyms).intersection(set(words))

def contain_currency_name_around_num(sentence, num_indices, window_width=4):
  # Whether words at the right side of NUM contain one of synonyms. 
  # (e.g. '30 dollars')
  words = common.flatten([sentence[idx+1:idx+1+window_width] 
                          for idx in num_indices])
  return set(c_names).intersection(set(words))

def contain_currency_symbol_around_num(sentence, num_indices, window_width=4):
  words = common.flatten([sentence[max(0, idx-window_width):idx] 
                          for idx in num_indices])
  return set(c_symbols).intersection(set(words))

@common.timewatch()
def main(args):
  assert re.match('(.+)\.normalized', args.input_file)
  dic = collections.OrderedDict()
  cnt = 0
  n_same_sentences = 0
  sys.stderr.write("Filtering %s...\n" % args.input_file)
  normalized_sentences = [l.replace('\n', '') for l in open(args.input_file)]
  for i, l in enumerate(normalized_sentences):
    cnt += 1
    save_l = l
    if save_l in dic:
      n_same_sentences += 1
      continue
    else:
      dic[save_l] = -1
    if args.max_lines and i > args.max_lines:
      break
    l = [wnl.lemmatize(x.lower()) for x in l.replace('\n', '').split(' ')]
    num_indices = [j for j, x in enumerate(l) if x == NUM.lower()]
    res1 = contain_synonym_around_num(l, num_indices, 
                                      window_width=args.window_width)
    res2 = contain_currency_name_around_num(l, num_indices,
                                            window_width=args.window_width)
    res3 = contain_currency_symbol_around_num(l, num_indices,
                                              window_width=args.window_width)
    if res1 and (res2 or res3):
      dic[save_l] = i

  acceptable_idx = [x for x in dic.values() if x != -1]
  random.shuffle(acceptable_idx)

  original_path = re.match('(.+)\.normalized', args.input_file).group(1)
  original_sentences = [l.replace('\n', '') for l in open(original_path)]
  with open(original_path + '.strict.idx', 'w') as f:
    sys.stdout = f
    for i in acceptable_idx:
      print i
    sys.stdout = sys.__stdout__

  with open(original_path + '.strict.origin', 'w') as f:
    sys.stdout = f
    for i in acceptable_idx:
      print original_sentences[i]
    sys.stdout = sys.__stdout__

  with open(args.input_file + '.strict', 'w') as f:
    sys.stdout = f
    for i in acceptable_idx:
      print normalized_sentences[i]
    sys.stdout = sys.__stdout__

  pos = [l.replace('\n', '') for l in open(args.input_file + '.pos')]
  print len(original_sentences), len(normalized_sentences), len(pos)
  with open(args.input_file + '.strict.pos', 'w') as f:
    sys.stdout = f
    for i in acceptable_idx:
      print pos[i]
    sys.stdout = sys.__stdout__


  sys.stderr.write("Obtain %d unique results from %d possible sentences. (%d same sentences)" % (len(acceptable_idx), cnt, n_same_sentences))

if __name__ == "__main__":
  desc = 'This script requires: f, f.normalized, f.normalized.pos'
  #parser = argparse.ArgumentParser(desc=desc)
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", 
                      default="results/candidate_sentences/all.normalized",
                      type=str, help="")
  parser.add_argument("-m", "--max_lines", default=0,
                      type=int, help="")
  parser.add_argument("-w", "--window_width", default=4,
                      type=int, help="")

  args  = parser.parse_args()
  main(args)

