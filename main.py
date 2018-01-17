# coding:utf-8 
import sys, re, argparse, time, commands, os
import spacy #, sense2vec
import utils
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from currency import get_currency_tokens
from nltk.tag.perceptron import PerceptronTagger
from nltk.tag.stanford import StanfordPOSTagger
#from nltk.tag.stanford import CoreNLPPOSTagger as StanfordPOSTagger

#python -m spacy download en
nlp = spacy.load('en_core_web_sm')
wnl = WordNetLemmatizer()
nltk_tagger = PerceptronTagger()

TAGGER_DIR = '/home/shoetsu/downloads/stanford-postagger'
stanford_tagger = StanfordPOSTagger(
  TAGGER_DIR + '/models/english-left3words-distsim.tagger',
  TAGGER_DIR + '/stanford-postagger.jar'
)

NUM = "__NUM__"
TMP_DIR = '/tmp/extractor_tmp'
logger = utils.logManager()

#########################################
##    Functions for simple filtering
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



# def include_numeric_spacy(sentence): # Stop using Spacy due to its slowness.
#   NUM = "NUM"
#   assert isinstance(sentence, spacy.tokens.doc.Doc)
#   #res = [i for i, span in enumerate(sentence) if span.pos_ == NUM]
#   res = [span for i, span in enumerate(sentence) if span.pos_ == NUM]
#   return res

# def include_numeric_manually(sentence):
#   numeric_patterns = [
#     '[0-9.,]*[0-9]',
#     'one','two','three','four', 'five', 'six', 'seven'
#   ]
#   return sentence

def include_numeric(sentence):
  res = [tok for tok, pos in stanford_tagger.tag(sentence) if pos == u'CD']
  return res

def find_sents_with_numerics(tokenized_sentences, tmp_filename=None):
  if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
  if tmp_filename:
    tmp_filepath = os.path.join(TMP_DIR, tmp_filename)
  else:
    tmp_filename = utils.random_string(5)
    tmp_filepath = os.path.join(TMP_DIR, tmp_filename)
    with open(tmp_filepath, 'w') as f:
      for l in tokenized_sentences:
        line = ' '.join(l) + '\n'
        assert unicode not in line.split(' ')
        f.write(line)

  if not os.path.exists(tmp_filepath + '.tagged'):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stanford-postagger.sh')
    logger.info('Running POS analysis to %d sentences...' % len(tokenized_sentences))
    cmd = "%s %s" % (script_path, tmp_filepath) 
    os.system(cmd)

    logger.info('The results of POS tagging is written in \'%s\'' % (tmp_filepath + '.tagged'))

  pos_tags = commands.getoutput('cut -f2 %s' % (tmp_filepath + '.tagged')).split('\n\n')
  res = [True if 'CD' in t.split('\n') else False for t in pos_tags]
  #os.system('rm %s' % tmp_filepath)
  #os.system('rm %s' % tmp_filepath + '.tagged')
  return tmp_filepath, res


def extract_sentences(input_texts, tmp_filename=None):
  candidates = []
  tokenized_candidates = []
  synonyms = set([
    'amount', 'bill', 'cost', 'demand', 'discount', 'estimate', 'expenditure', 'expense', 'fare', 'fee', 'figure', 'output', 'pay', 'payment', 'premium', 'rate', 'return', 'tariff', 'valuation', 'worth', 'appraisal', 'assessment', 'barter', 'bounty', 'ceiling', 'charge', 'compensation', 'consideration', 'damage', 'disbursement', 'dues', 'exaction', 'hire', 'outlay', 'prize', 'quotation', 'ransom', 'reckoning', 'retail', 'reward', 'score', 'sticker', 'tab', 'ticket', 'toll', 'tune', 'wages', 'wholesale', 'appraisement', 
  ])
  currency_symbols, currency_names = get_currency_tokens()
  currency_symbols = [c.replace('$', '\$') for c in currency_symbols] # for regexp

  def find_shared_token(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    return s1.intersection(s2)

  def find_shared_pattern(sentence, exprs):
    res = []
    sentence = " ".join(sentence)
    for expr in exprs:
      m = re.search(expr, sentence)
      if m:
        res.append(m.group(0))
    return res

  # list of synonyms given from http://www.thesaurus.com/browse/price?s=t
  t_lemmatize, t_currency, t_synonym, t_spacy, t_numeric = 0,0,0,0,0
  logger.info("Filtering sentences by whether they contain synonyms of 'price' (charge, cost), currency units (dollar, franc), or currency symbols($, ₣)...")
  t_all = time.time()
  for i, line in enumerate(input_texts):
    if tmp_filename and os.path.exists(tmp_filename):
      break
    if i and i % 100000 == 0:
      logger.info('Done %d/%d ... (%f sec per a line )' % (i, len(input_texts), (time.time() - t_all)/i ))
    line = line.encode('ascii', 'ignore')
    if not line:
      continue
    t = time.time()
    tokenized_text = [token.lower() for token in word_tokenize(line)]
    t_tokenize = time.time() - t
    t = time.time()
    lemmatized_text = [wnl.lemmatize(token) for token in tokenized_text]
    t_lemmatize += time.time() - t

    # Filter sentences by whether they contain synonyms of 'price' (charge, cost), currency units (dollar, franc), or currency symbols($, ₣).
    t = time.time()
    res_synonym = find_shared_token(lemmatized_text, synonyms)
    t_synonym += time.time() - t

    t = time.time()
    res_currency1 = find_shared_token(lemmatized_text, currency_names)
    res_currency2 = find_shared_pattern(lemmatized_text, currency_symbols)
    res_currency = set(res_currency1).union(set(res_currency2))
    t_currency += time.time() - t

    if not (res_synonym or res_currency):
      continue

    # t = time.time()
    # doc = nlp(line)
    # t_spacy += time.time() - t

    t = time.time()
    
    ## Changed: as calling POS tagger every time this processes a new line can be very costful, this creates a temporary file of possible sentences and they are collectively processed with POS tagger.
    #res_numeric = include_numeric(doc)
    #res_numeric = include_numeric(tokenized_text)
    #res_numeric = include_numeric_manually(tokenized_text)
    #if not res_numeric:
    #  continue
    #t_numeric += time.time() - t
    candidates.append(line)
    tokenized_candidates.append(tokenized_text)

  if tmp_filename:
    

  t = time.time()
  _, contains_numeric = find_sents_with_numerics(tokenized_candidates, tmp_filename)
  print len(contains_numeric), len(tokenized_candidates)
  assert len(contains_numeric) == len(tokenized_candidates)
  candidates = [s for x, s in zip(contains_numeric, candidates) if x]
  tokenized_candidates = [" ".join(s) for x, s in zip(contains_numeric, tokenized_candidates) if x]
  t_numeric += time.time() - t

  #results = set([re.sub("[0-9.,]*[0-9]", NUM, sent) for sent in tokenized_candidates])
  #results = set([re.sub("[0-9.,]*[0-9]", NUM, sent) for sent in candidates])
  results = candidates

  t_all = time.time() - t_all
  sys.stdout = sys.stderr
  print ""
  print "<Time spent>"
  print "Total: %f  (%f per a sent)" % (t_all, t_all / i)
  print "Tokenize: %.1f %%" % (t_tokenize / t_all * 100)
  print "Lemmatize: %.1f %%" % (t_lemmatize / t_all * 100)
  print "Synonyms: %.1f %%" % (t_synonym / t_all * 100)
  print "Currency: %.1f %%" % (t_currency / t_all * 100)
  #print "Creating Spacy: %.1f %%" % (t_spacy / t_all * 100)
  print "Numeric: %.1f %%" % (t_numeric / t_all * 100)
  sys.stdout = sys.__stdout__
  
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
  return [l for l in res if l and apply_restriction(l, restrictions)]


def debug():
  
  texts =  [
    u'This costs at least 1,000 dollars.' ,
    u'This costs at least thirty-eight dollars.' ,
    u'This costs at least ￥1,000',
    u'9 Responses to “Real or fake Christmas tree?”'
  ] 
  for text in texts:
    text = word_tokenize(text)
    print text
    print stanford_tagger.tag(text)
  exit(1)
  # Occasionally Spacy fails to parse unicode token.
  def print_pos_lemma(doc):
    if not isinstance(doc, spacy.tokens.doc.Doc):
      doc = nlp(doc)
    for t in doc:
      print t, t.pos_, t.lemma_
      print ""

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
  input_file = args.input_file

  with open(input_file, "r",) as ifile:
    input_texts = ifile.read().decode('utf-8')
    if args.max_lines:
      input_texts = input_texts[:args.max_lines]

    # Reduce the number of candidate sentences by simple regexps.
    include_no_noisy_tokens = lambda x: True if not re.search("[{};=\|]", x) else False # for some reason, the stanford parser judges parentheses as numerical values
    include_number_f = lambda x: True if re.search("[0-9.,]*[0-9]", x, re.I) else False
    include_dollar_f = lambda x: True if re.search("\$[0-9.,]*[0-9]", x, re.I) else False

    restrictions = [include_no_noisy_tokens]
    n_original = len(input_texts)
    if not(args.tmp_file):
      input_texts = preprocess(input_texts, restrictions=restrictions)
      n_preprocess = len(input_texts)
      results = extract_sentences(input_texts, args.tmp_file)
      logger.info("Number of lines in original data: {}".format(n_original))
      logger.info("Number of lines after preprocessing: {}".format(n_preprocess))
  logger.info("Number of entries: {}".format(len(results)))
  print "\n".join(results)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", default="small.txt",
                      type=str, help="")
  parser.add_argument("-m", "--max_lines", default=0,
                      type=int, help="")
  parser.add_argument("-t", "--tmp_file", default=None,
                      type=str, help="")

  args  = parser.parse_args()
  main(args)
