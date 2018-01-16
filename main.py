# coding:utf-8 
import sys, re, argparse, time
import spacy #, sense2vec
import utils
from currency import get_currency_tokens

#python -m spacy download en
nlp = spacy.load('en_core_web_sm')

# successes : L54, L200, L249
# failures  : L90, L236, L271

#########################################
##    Functions for simple filtering
#########################################

def include_number(sentence):
  assert isinstance(sentence, spacy.tokens.doc.Doc)
  m = re.search(u'[0-9]', sentence.text)
  return True if m else False

def include_numeric(sentence):
  NUM = "NUM"
  assert isinstance(sentence, spacy.tokens.doc.Doc)
  #res = [i for i, span in enumerate(sentence) if span.pos_ == NUM]
  res = [span for i, span in enumerate(sentence) if span.pos_ == NUM]
  
  return res

def find_sharing_token(sentence, tokens, lemmatize=True):
  assert isinstance(sentence, spacy.tokens.doc.Doc)
  tokens_in_sent = set([t.lemma_ for t in sentence]) if lemmatize else set([t.text for t in sentence])
  return tokens_in_sent.intersection(tokens)
#########################################


def extract_expression(sentence):
  assert isinstance(sentence, spacy.tokens.doc.Doc)

  def pos_pattern_based(sentence):
    """
    """
    # TODO: compare with regular expressions of POS, not direct matching.
    pos_patterns = [
      #[u'NUM', 'NOUN'], # fifty dollars
      [u'SYM', u'NUM'], # $100

      [u'SYM', u'NUM', u'PART', u'SYM', u'NUM'], # $5.95 to $8.75
      [u'SYM', u'NUM', u'SYM'], # $100+
      [u'PROPN', u'SYM', u'NUM'], # Value $40.00
      [u'NOUN',  u'SYM', u'NUM'], # Cost $1.9 
      [u'NOUN', u'PUNCT', u'SYM', u'NUM'], #Cost: $1.9 
      [u'NOUN', u'PUNCT', u'SYM', u'NUM', u'NUM'], #Cost: $1.9 million
      [u'SYM', u'NUM', u'SYM', u'SYM', u'NUM'], #$300 - $500
      [u'ADV', u'SYM', u'NUM'], # only $24.95
      [u'ADV', u'SYM', u'NUM', u'ADJ'], # only $5.00 more
      [u'SYM', u'NUM', u'CCONJ', u'ADP', u'NOUN'], # 100,000+ per year
      [u'SYM', u'NUM', u'DET'], # $5.00 each
      [u'ADV', u'ADV', u'SYM', u'NUM'] # at least $20
    ]

    pos_of_sent = [t.pos_ for t in sentence]
    idx_expression = []
    for pos_pattern in pos_patterns:
      for i in xrange(len(pos_of_sent) - len(pos_pattern) + 1):
        if pos_of_sent[i:i+len(pos_pattern)] == pos_pattern:
          idx_expression.append([j for j in xrange(i, i+len(pos_pattern))])
    return idx_expression

  def pos_regexp_based(sentence):
    idx_expression = []
    return idx_expression

  extract_f = pos_pattern_based
  return extract_f(sentence)


@utils.timewatch()
def extract_sentences(input_texts):
  results = []
  currencies = get_currency_tokens(lemmatize=args.lemmatize)

  # list of synonyms given from http://www.thesaurus.com/browse/price?s=t
  synonyms = set([
    'amount', 'bill', 'cost', 'demand', 'discount', 'estimate', 'expenditure', 'expense', 'fare', 'fee', 'figure', 'output', 'pay', 'payment', 'premium', 'rate', 'return', 'tariff', 'valuation', 'worth', 'appraisal', 'assessment', 'barter', 'bounty', 'ceiling', 'charge', 'compensation', 'consideration', 'damage', 'disbursement', 'dues', 'exaction', 'hire', 'outlay', 'prize', 'quotation', 'ransom', 'reckoning', 'retail', 'reward', 'score', 'sticker', 'tab', 'ticket', 'toll', 'tune', 'wages', 'wholesale', 'appraisement', 
    #'asking price', 'face value',  # MWEs are ignored for now.
  ])
  t0, t1, t2, t3, t4 = 0,0,0,0,0
  t_all = time.time()
  for i, line in enumerate(input_texts):
    if i > args.max_lines:
      break

    if not line:
      continue
    t = time.time()
    doc = nlp(line)
    t0 += time.time() - t

    t = time.time()
    res_numeric = include_numeric(doc)
    t1 += time.time() - t

    t = time.time()
    if not res_numeric:
      continue
    res_currency = find_sharing_token(doc, currencies, lemmatize=args.lemmatize)
    t2 += time.time() - t

    t = time.time()
    res_synonyms = find_sharing_token(doc, synonyms, lemmatize=True)
    t3 += time.time() - t

    if not (res_synonyms or res_currency):
      continue
    print "<L%d>\t%s" % (i, line)
    print res_currency.union(res_synonyms).union(set(res_numeric))
    results.append(line)

  t_all = time.time() - t_all
  print ""
  print "<Time spent>"
  print "Total", t_all
  print "Creating Spacy", t0
  print "Numeric", t1
  print "Synonyms", t2
  print "Currency", t3
  return results


@utils.timewatch()
def extract(input_texts): # Deprecated
  # Codes for expression extraction (this is to be done after clustering?)
  ins_count = 0
  showed_list = []
  idx_expression = extract_expression(doc)
  if idx_expression and idx_expression not in showed_list:
    print "<L%d>\t" % i
    flattened_indice = list(set(utils.flatten(idx_expression)))
    print 'Original sentence:\t',
    utils.print_colored([t.text for t in doc], flattened_indice, 'red')
    print 'POS list         :\t',
    utils.print_colored([t.pos_ for t in doc], flattened_indice, 'blue')
    print 'Expressions      :\t',
    print [(" ".join([doc[i].text for i in indices]), indices[0], indices[-1]) for indices in idx_expression]
    showed_list.append(idx_expression)
    ins_count += 1
  return ins_count



@utils.timewatch()
def preprocess(input_texts, restrictions=[lambda x: True if x else False]):
  """
  <Args>
  input_texts: list of unicode string.
  restrictions: list of functions to decide whether a line is acceptable or not.
  """

  def apply_restriction(l, functions):
    for f in functions:
      if not f(l):
        return False
    return True

  res = [l.strip() for l in re.sub('[ \t]+', " ", input_texts).split('\n')]
  return [l for l in res if apply_restriction(l, restrictions)]


def print_pos_lemma(doc):
  if not isinstance(doc, spacy.tokens.doc.Doc):
    doc = nlp(doc)
  for t in doc:
    print t, t.pos_, t.lemma_
  print ""
  return

def debug():
  # Occasionally Spacy fails to parse unicode token.
  text = u'This costs at least ￥1,000' 
  print_pos_lemma(text)
  """
  This DET this
  costs VERB cost
  at ADV at
  least ADJ least
  ￥1,000 ADJ ￥1,000
  """

  text = u'This costs at least $1,000' 
  """
  This DET this
  costs VERB cost
  at ADV at
  least ADV least
  $ SYM $
  1,000 NUM 1,000
  """
  print_pos_lemma(text)


@utils.timewatch()
def main(args):
  #debug()
  input_file = args.input_file
  with open(input_file, "r",) as ifile:
    input_texts = ifile.read().decode('utf-8')

    # Reduce the number of candidate sentences by simple regexps.
    include_no_noisy_tokens = lambda x: True if not re.search("[;=]", x) else False
    include_number_f = lambda x: True if re.search("[0-9.,]+[0-9]", x, re.I) else False
    include_dollar_f = lambda x: True if re.search("\$[0-9.,]+[0-9]", x, re.I) else False

    restrictions = [include_no_noisy_tokens]
    input_texts = preprocess(input_texts, restrictions=restrictions)
    res = extract_sentences(input_texts)
    print("Number of entries: {}".format(len(res)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", default="small.txt",
                      type=str, help="")
  parser.add_argument("-l", "--lemmatize", default=True,
                      type=utils.str2bool, help="")
  parser.add_argument("-m", "--max_lines", default=30000,
                      type=int, help="")
  #parser.add_argument("-o", "--output_file", default="results.txt", 
  #                    type=str, help="")
  args  = parser.parse_args()
  main(args)
