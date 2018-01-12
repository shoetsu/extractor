# coding:utf-8 
import sys, re, argparse
import spacy #, sense2vec
import utils

#python -m spacy download en
nlp = spacy.load('en_core_web_sm')


# successes : L54, L200, L249
# failures  : L90, L236, L271

def include_numeric(sentence):
  NUM = "NUM"
  assert isinstance(sentence, spacy.tokens.doc.Doc)
  pos = [i for i, span in enumerate(sentence) if span.pos_ == NUM]
  return pos

def include_number(sentence):
  m = re.search(u'[0-9]', sentence.text)
  return True if m else False

def print_pos(sentence):
  for s in sentence:
    print s, s.pos_

def print_colored(sentence, idx_colored, color='red'):
  #assert isinstance(sentence, spacy.tokens.doc.Doc)
  assert isinstance(sentence, list)
  res = []
  for i,s in enumerate(sentence):
    if i in idx_colored:
      res.append(utils.colored(s, color))
    else:
      res.append(s)
  print " ".join(res)

def extract_expression(sentence):
  assert isinstance(sentence, spacy.tokens.doc.Doc)

  def pos_based(sentence):
    """
    """
    # TODO: compare by regular expressions of POS, not direct matching.
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

  extract_f = pos_based
  return extract_f(sentence)

def debug():
  pass

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
      print_colored([t.text for t in doc], flattened_indice, 'red')
      print 'POS list         :\t',
      print_colored([t.pos_ for t in doc], flattened_indice, 'blue')
      print 'Expressions      :\t',
      print [" ".join([doc[i].text for i in indices]) for indices in idx_expression]
      showed_list.append(idx_expression)
      ins_count += 1
  return ins_count


@utils.timewatch()
def preprocess(input_texts, reg_exp=None):
  res = [l.strip() for l in re.sub('[ \t]+', " ", input_texts).split('\n')]
  if reg_exp:
    return [l for l in res if re.search(reg_exp, l)]
  else:
    return [l for l in res if l]

def main(args):
  input_file = args.input_file
  with open(input_file, "r",) as ifile:
    input_texts = ifile.read().decode('utf-8')
    #reg_exp = "[0-9.,]+[0-9]"
    #reg_exp = "(\$[0-9.,]+[0-9])|([0-9.,]+[0-9]\s+dollar(s)?)"
    reg_exp = None
    input_texts = preprocess(input_texts, reg_exp=reg_exp)
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
