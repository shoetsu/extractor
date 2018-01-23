#coding: utf-8
import sys, re, argparse, time, commands, os
import spacy #, sense2vec
from utils import common, currency
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
NUM = common.NUM
c_symbols, c_names = currency.get_currency_tokens()

def contain_synonym_around_num(sentence, num_indices, window_width=3):
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
  synonyms = ['price', 'toll', 'cost', 'pay', 'worth', 'sell', 'charge']
  # Whether words at the left side of NUM contain one of synonyms. 
  # (e.g. 'cost $ 30')
  words = common.flatten([sentence[min(0, idx-window_width):idx] 
                          for idx in num_indices])
  return set(synonyms).intersection(set(words))

def contain_currency_name_around_num(sentence, num_indices, window_width=3):
  # Whether words at the right side of NUM contain one of synonyms. 
  # (e.g. '30 dollars')
  words = common.flatten([sentence[idx+1:idx+1+window_width] 
                          for idx in num_indices])
  return set(c_names).intersection(set(words))

def contain_currency_symbol_around_num(sentence, num_indices, window_width=3):
  words = common.flatten([sentence[min(0, idx-window_width):idx] 
                          for idx in num_indices])
  return set(c_symbols).intersection(set(words))

@common.timewatch()
def main(args):
  for i, l in enumerate(open(args.input_file)):
    original_l = l.replace('\n', '')
    if args.max_lines and i > args.max_lines:
      break
    l = [wnl.lemmatize(x.lower()) for x in l.replace('\n', '').split(' ')]
    num_indices = [j for j, x in enumerate(l) if x == NUM.lower()]
    res1 = contain_synonym_around_num(l, num_indices)
    res2 = contain_currency_name_around_num(l, num_indices)
    res3 = contain_currency_symbol_around_num(l, num_indices)
    if res1 or res2 or res3:
      #print '<L%d>' % i
      print original_l
      #print res1.union(res2).union(res3)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", 
                      default="results/candidate_sentences/all.normalized.txt",
                      type=str, help="")
  parser.add_argument("-m", "--max_lines", default=0,
                      type=int, help="")

  args  = parser.parse_args()
  main(args)

