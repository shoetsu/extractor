import numpy as np
import multiprocessing as mp
import time, random, os, commands, collections, re
from datetime import datetime
from itertools import chain
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL
try:
  import cPickle as pickle
except:
  import pickle
############################################
def load_or_create(processed_path, func, *args):
  print (processed_path)
  if not os.path.exists(processed_path):
    data = func(*args)
    pickle.dump(data, open(processed_path, 'wb'))
  else:
    data = pickle.load(open(processed_path, "rb"))
  return data



####################
#   Constants
####################
NUM = "__NUM__"
NUMBER = "703" # in parsing, "__NUM__" can make troubles.
UNK = "__UNK__" 
TAGGER_DIR = '/home/shoetsu/downloads/stanford-postagger'
####################


def quote(str_):
  return "\"" + str_ + "\""

def timestamp():
  return datetime.now()

def random_string(length, seq='0123456789abcdefghijklmnopqrstuvwxyz'):
    sr = random.SystemRandom()
    return ''.join([sr.choice(seq) for i in xrange(length)])

def restore_to_tmpfile(sentences, tmp_dir='/tmp'):
  """
  sentences: List of str.
  """
  tmp_filename = random_string(5)
  tmp_filepath = os.path.join(tmp_dir, tmp_filename)
  with open(tmp_filepath, 'w') as f:
    for line in sentences:
      f.write(line + '\n')
  return tmp_filepath


def str2tuple(v, type_f=int): 
  if type(v) in [list, tuple]:
    res = (type_f(x) for x in v if x != ',')
  else:
    m = re.match('\((.+)\)', v)
    if m:
      v = m.group(1)
    res = (type_f(x) for x in v.split(','))
  return tuple(list(res))

def str2bool(v):
  if type(v) == bool:
    return v
  return v.lower() in ("yes", "true", "t", "1")

def flatten(l):
  return list(chain.from_iterable(l))


def logManager(logger_name='main', 
              handler=StreamHandler(),
              log_format = "[%(levelname)s] %(asctime)s - %(message)s",
              level=DEBUG):
    formatter = Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = getLogger(logger_name)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# def timewatch(func, logger=None):
#   if logger is None:
#     logger = logManager(logger_name='utils')
#   def _wrapper(*args, **kwargs):
#     start = time.time()
#     result = func(*args, **kwargs)
#     end = time.time()
#     logger.info("%s: %f sec" % (func.__name__ , end - start))
#     return result
#   return _wrapper

def timewatch(logger=None):
    if logger is None:
      logger = logManager(logger_name='utils')
    def _timewatch(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info("%s: %f sec" % (func.__name__ , end - start))
            return result
        return wrapper
    return _timewatch

def print_colored(sentence, idx_colored, color='red'):
  assert isinstance(sentence, list)
  res = []
  for i,s in enumerate(sentence):
    if i in idx_colored:
      res.append(colored(s, color))
    else:
      res.append(s)
  print " ".join(res)

def colored(str_, color):
  '''
  Args: colors: a str or list of it.
  '''
  RESET = "\033[0m"
  ctable = {
    'black': "\033[30m",
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\033[33m",
    'blue': "\033[34m",
    'purple': "\033[35m",
    'underline': '\033[4m',
    'link': "\033[31m" + '\033[4m',
    'bold': '\033[30m' + "\033[1m",
  }
  if type(color) == str:
    res = ctable[color] + str_ + RESET
  elif type(color) == tuple or type(color) == list:
    res = "".join([ctable[c] for c in color]) + str_ + RESET
  return res 


class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

class recDotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  def __init__(self, _dict):
    for k in _dict:
      if isinstance(_dict[k], dict):
        _dict[k] = recDotDict(_dict[k])
      # if isinstance(_dict[k], list):
      #    for i,x in enumerate(_dict[k]):
      #      print i, x, isinstance(x, dict)
      #      if isinstance(x, dict):
      #        dict[k][i] = dotDict(x)
    super(recDotDict, self).__init__(_dict)


def multi_process(func, *args):
  '''
  Args:
    - func : a function to be executed.
    - args : a list of list of args that a worker needs.
             [[id1, name1, ...], [id2, name2, ...]]
  '''
  # A wrapper to make a function put its response to a queue.
  def wrapper(_func, idx, q):
    def _wrapper(*args, **kwargs):
      res = func(*args, **kwargs)
      return q.put((idx, res))
    return _wrapper

  workers = []
  # mp.Queue() seems to have a bug..?
  # (stackoverflow.com/questions/13649625/multiprocessing-in-python-blocked)
  q = mp.Manager().Queue()

  # kwargs are not supported... (todo)
  for i, a in enumerate(zip(*args)):
    worker = mp.Process(target=wrapper(func, i, q), args=a)
    workers.append(worker)
    worker.daemon = True  # make interrupting the process with ctrl+c easier
    worker.start()

  for worker in workers:
    worker.join()
  results = []
  while not q.empty():
    res = q.get()
    results.append(res)

  return [res for i, res in sorted(results, key=lambda x: x[0])]


def get_ngram_match(sent, ngram):
  if type(sent) == str:
    sent = sent.split(' ')
  n = len(ngram)
  # print sent, ngram
  # for i in xrange(len(sent)-n+1):
  #   print tuple(sent[i:i+n])
  indices = [(i, i+n-1) for i in xrange(len(sent)-n+1) if tuple(sent[i:i+n]) == ngram]
  return indices

def get_ngram(s, min_n, max_n, vocab_condition=lambda x: True):
  return [[tuple(s[i:i+n]) for i in xrange(len(s)-n+1) if vocab_condition(s[i:i+n])] for n in xrange(min_n, max_n+1)]

def check_overlaps(existing_spans, new_spans):
  # Input : list of tuples [(int, int), ....]
  res = []
  for ns in new_spans:
    check = [True if es[1] < ns[0] and ns[1] < es[0] else False for es in existing_spans]
    if not False in check:
      res.append(ns)
  return res

def tokenize_heuristics(sent):
  # for some reason nltk.tokenizer fails to separate numbers (e.g. 6.73you)
  for m in re.findall("([0-9][0-9\,\.]*)[a-zA-Z]+",sent):
    sent = sent.replace(m, m + ' ')
  return sent

def unzip(l):
    #*map(list, zip(sents, sents_pos))
  return map(list, zip(*l))


def ask_yn(message, func, *args, **kwargs):
  print message
  print func
  print args
  print kwargs
  while True:
    print message
    #print "Remove the old results? [Y/n] (%s)" % output_dir
    x = raw_input().lower()
    if x == 'y' or x == '\n' or x == '':
      func(*args, **kwargs)
      #os.system('rm -r %s' % output_dir)
      break
    elif x == 'n':
      print "Operation was aborted."
      exit(1)
    else:
      print "Type 'y' or 'n'."
