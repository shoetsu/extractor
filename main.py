# coding:utf-8 
import sys, re, argparse
import spacy #, sense2vec
import utils

#python -m spacy download en
nlp = spacy.load('en_core_web_sm')

# successes : L54, L200, L249
# failures  : L90, L236, L271


#########################################
##    Functions for spacy object
#########################################

def include_numeric(sentence):
  NUM = "NUM"
  assert isinstance(sentence, spacy.tokens.doc.Doc)
  pos = [i for i, span in enumerate(sentence) if span.pos_ == NUM]
  return pos

def include_number(sentence):
  assert isinstance(sentence, spacy.tokens.doc.Doc)
  m = re.search(u'[0-9]', sentence.text)
  return True if m else False

def print_pos(sentence):
  assert isinstance(sentence, spacy.tokens.doc.Doc)
  for s in sentence:
    print s, s.pos_

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
      [u'PUNCT', u'SYM', u'NUM', u'PUNCT'], # - $75 -
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

  extract_f = pos_pattern_based
  return extract_f(sentence)

@utils.timewatch()
def extract(input_texts, output_file=None):
  ins_count = 0
  num_numeric = 0
  num_no_number = 0

  res = []
  showed_list = []
  for i, line in enumerate(input_texts):
    if not line:
      continue
    doc = nlp(line)
    idx_numeric = include_numeric(doc)
    if not idx_numeric:
      continue
    num_numeric += 1
    res.append(doc)
    if not include_number(doc):
      num_no_number += 1

    idx_expression = extract_expression(doc)
    if idx_expression and idx_expression not in showed_list:
      print "<L%d>\t" % i
      flattened_indice = list(set(utils.flatten(idx_expression)))
      print 'Original sentence:\t',
      utils.print_colored([t.text for t in doc], flattened_indice, 'red')
      print 'POS list         :\t',
      utils.print_colored([t.pos_ for t in doc], flattened_indice, 'blue')
      print 'Expressions      :\t',
      print [" ".join([doc[i].text for i in indices]) for indices in idx_expression]
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

def debug():
  pass

def main(args):
  input_file = args.input_file
  with open(input_file, "r",) as ifile:
    input_texts = ifile.read().decode('utf-8')

    # Reduce the number of candidate sentences by simple regexps.
    include_no_noisy_tokens = lambda x: True if not re.search("[;=]", x) else False
    include_number_f = lambda x: True if re.search("[0-9.,]+[0-9]", x) else False
    include_dollar_f = lambda x: True if re.search("\$[0-9.,]+[0-9]", x) else False
    #restrictions = [include_no_noisy_tokens, include_number_f]
    restrictions = [include_no_noisy_tokens]
    input_texts = preprocess(input_texts, restrictions=restrictions)
    ins_count = extract(input_texts)
    print("Number of entries: {}".format(ins_count))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", default="small.txt",
                      type=str, help="")
  #parser.add_argument("-o", "--output_file", default="results.txt", 
  #                    type=str, help="")
  args  = parser.parse_args()
  main(args)
