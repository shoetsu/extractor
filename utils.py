import time, random
from itertools import chain
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL

def random_string(length, seq='0123456789abcdefghijklmnopqrstuvwxyz'):
    sr = random.SystemRandom()
    return ''.join([sr.choice(seq) for i in xrange(length)])

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
