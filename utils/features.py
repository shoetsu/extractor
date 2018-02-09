#coding:utf-8
import numpy as np
import spacy
import time, random, os, commands, collections, re, sys
from utils import common

try:
  import cPickle as pickle
except:
  import pickle

NUM = common.NUM
NUMBER = common.NUMBER

######################################
#  Visualizing dependencies on Spacy
from nltk import Tree
def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])

def print_tree(doc):
  if type(doc) == spacy.tokens.span.Span:
    sent = doc
    return to_nltk_tree(sent.root).pretty_print()
  else:
    return [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)
#######################################

class Vectorizer(object):
  def __init__(self, *args, **kwargs):
    self.vocab = None
    self.rev_vocab = None

  @property
  def size(self):
    return len(self.vocab) if self.vocab else None

  def fit(self, *args, **kwargs):
    return self.fit_transform(*args, **kwargs)

  def fit_transform(self, *args, **kwargs):
    raise NotImplementedError

class MultiVectorizerWrapper(object):
  def __init__(self, vectorizers, *args, **kwargs):
    self.vectorizers = vectorizers

  def vec2tokens(self, vectors):
    tokens = []
    current_dim = 0
    for v in self.vectorizers:
      size = v.size
      vecs = vectors[:, current_dim:current_dim+size]
      current_dim += size
      tokens.append(v.vec2tokens(vecs))
    res = [common.flatten(t) for t in zip(*tokens)]
    return res 

  def fit(self, *args, **kwargs):
    return self.fit_transform(*args, **kwargs)

  def fit_transform(self, *args, **kwargs):
    features = []
    for v in self.vectorizers:
      indices, f = v.fit(*args, **kwargs) 
      features.append(f)
    features = np.concatenate(features, axis=-1)
    #features = [v.fit(*args, **kwargs) for v in self.vectorizers]
    return indices, features

class NGramVectorizer(Vectorizer):
  def __init__(self, idx, output_dir, ngram_range=(1,2), min_freq=0, 
               vocab_size=0, vocab_condition=lambda x: True):
    #super(NGramVectorizer, self).__init__()
    assert ngram_range[0] > 0 and len(ngram_range) == 2 
    assert type(min_freq) == int
    """
    Setup its vocabulary when fit_transform is called first.
    """
    super(NGramVectorizer, self).__init__()
    self.idx = idx
    self.output_dir = output_dir
    self.ngram_range = ngram_range
    self.min_freq = min_freq
    self.vocab_condition = vocab_condition
    self.vocab_path = output_dir + '/cluster.vocab'
    if idx > 0:
      self.vocab_path += '.%d' % (idx)
    self._load_vocab()
    self.vocab_size = vocab_size

  def _save_vocab(self):
    if not self.vocab:
      raise Exception('This vectorizer has no vocabulary.')
    if not os.path.exists(self.vocab_path):
      with open(self.vocab_path, 'w') as f:
        for ng in self.vocab:
          f.write(' '.join(ng) + '\n')

  def _load_vocab(self):
    if os.path.exists(self.vocab_path):
      with open(self.vocab_path) as f:
        self.vocab = [tuple(l.replace('\n', '').split(' ')) for l in f]
        self.rev_vocab = collections.OrderedDict([(v, i) for i,v in enumerate(self.vocab)])

  def create_vocab(self, ngrams):
    min_freq = self.min_freq
    vocab = collections.Counter(common.flatten(ngrams))
    vocab = sorted([(v, vocab[v]) for v in vocab if not self.min_freq or vocab[v] >= self.min_freq], key=lambda x: -x[1])
    if self.vocab_size:
      vocab = vocab[:self.vocab_size]
    self.vocab = [v[0] for v in vocab]
    self.rev_vocab = collections.OrderedDict([(v, i) for i,v in enumerate(self.vocab)])
    self._save_vocab()

  def get_features(self, lines, input_filepath=None):
    return None, [common.flatten(common.get_ngram(s, self.ngram_range[0], self.ngram_range[1], vocab_condition=self.vocab_condition)) for s in lines]


  def fit_transform(self, lines, input_filepath=None):
    """
    lines: list of (tokenized) lines.
    """
    if type(lines[0]) == str:
      lines = [x.split(' ') for x in lines]

    indices, features = self.get_features(lines, input_filepath)
    if not self.vocab:
      self.create_vocab(features)

    feature_vectors = [np.bincount([self.rev_vocab[t] for t in collections.Counter(fv) if t in self.rev_vocab], minlength=len(self.vocab)) for fv in features]

    # normalize vectors
    feature_vectors = np.asarray([v / np.linalg.norm(v) if np.linalg.norm(v) else v for v in feature_vectors])
    return indices, feature_vectors

  def vec2tokens(self, vectors):
    assert type(vectors) == np.ndarray and len(vectors.shape) == 2 
    return [sorted([(self.vocab[idx], v[idx]) for idx in v.nonzero()[0] if self.vocab[idx] != (NUM,)], key=lambda x: -x[1]) for v in vectors] 

    # assert len(vectors.shape) <= 2
    # if len(vectors.shape) == 2:
    #   return [sorted([(self.vocab[idx], v[idx]) for idx in v.nonzero()[0] if self.vocab[idx] != (NUM,)], key=lambda x: -x[1]) for v in vectors] 
    # elif len(vectors.shape) == 1:
    #   v = vectors
    #   return sorted([(self.vocab[idx], v[idx]) for idx in v.nonzero()[0] if self.vocab[idx] != (NUM,)], key=lambda x: -x[1]) 


# @common.timewatch()
# def dependency_parsing(lines, input_filepath=None): #deprecated
#   """
#   input_filepath: if not None, the result of parsing is copied to 'input_filepath.dep'
#   """
#   if type(lines[0]) != str:
#     lines = [" ".join(s) for s in lines]
#   assert "" not in [s.strip() for s in lines]

#   if input_filepath and os.path.exists(input_filepath + '.dep'):
#     dep_path = input_filepath + '.dep'
#   else:
#     tmp_filepath = common.restore_to_tmpfile(lines, tmp_dir='/tmp/extractor_tmp')
#     cmd = './stanford-parser/lexparser.sh %s' % tmp_filepath
#     res = commands.getoutput(cmd)
#     if input_filepath:
#       os.system('cp %s %s' % (tmp_filepath + '.dep', input_filepath + '.dep'))
#     os.system('rm %s' % tmp_filepath)
#     dep_path = tmp_filepath + '.dep'

#   with open(dep_path) as f:
#     dependencies = [l.split('\n') for l in f.read().split('\n\n') if l]
#   return res


@common.timewatch()
def create_spacy(lines, input_filepath):

  output_path = input_filepath + '.spacy' if input_filepath else None
  docs = None
  if output_path and os.path.exists(output_path):
    sys.stderr.write('Loading Spacy file...\n')
    docs = pickle.load(open(output_path, 'rb'))
  else:
    sys.stderr.write('Creating new Spacy objects...\n')
    nlp = spacy.load('en_core_web_sm')
    assert type(lines) == list
    if type(lines[0]) == list:
      lines = [' '.join(l).decode('utf-8') for l in lines]
    lines = [l.replace(NUM, NUMBER) for l in lines]
    docs = [nlp(l) for l in lines]
    if output_path:
      pickle.dump(docs, open(output_path, 'wb'))
  return docs

class DependencyVectorizer(NGramVectorizer):
  def __init__(self, *args, **kwargs):
    super(DependencyVectorizer, self).__init__(*args, **kwargs)

  def trace(self, sent):
    def get_subtree(node):
      lefts = common.flatten([get_subtree(n) for n in node.lefts])
      rights = common.flatten([get_subtree(n) for n in node.rights])
      subtree = lefts + [node] + rights
      return subtree

    def trace_up_from_num(num_node, parents):
      #min_st=2, max_st=7
      min_st, max_st = self.ngram_range
      # Starting from NUM, trace upward with getting subtrees.
      node = num_node
      subtrees = []
      while True:
        st = get_subtree(node)
        if len(st) >= min_st and len(st) <= max_st and self.vocab_condition([t.string.strip().encode('utf-8') for t in st]): 
          subtrees.append(st)
        if node in parents:
          node = parents[node]
        else:
          break
      return [tuple(st) for st in subtrees]

      return res

    # Find NUM and the parent of each node.
    parents = {}
    num_nodes = []
    for token in sent:
      parents.update({c:token for c in token.children})

    res = []
    for i, token in enumerate(sent):
      if token.string.strip().encode('utf-8') == NUMBER: #NUMBER.decode('utf-8'):
        res.append((i, trace_up_from_num(token, parents)))
    return res

  # Using only the surface of each token in the subtree.
  def subtree2str(self, st):
    return tuple(" ".join([t.string.strip().encode('utf-8') for t in st]).replace(NUMBER, NUM).split(" "))

  def get_features(self, lines, input_filepath=None):
    docs = create_spacy(lines, input_filepath)
    indices = []
    features = []
    feature_f = self.subtree2str
    for i, d in enumerate(docs):
      # Get features per a NUM token in a line. # [(token_idx0, subtrees0), ...]
      offset = 0
      feature = []
      for s in d.sents:
        feature.append([(offset+idx, [feature_f(st) for st in sts]) for idx, sts in self.trace(s)])
        offset += len(s)
      feature = common.flatten(feature)
      idx, feature = zip(*feature) if feature else ((-1,), ([],))
      indices += [(i, j) for j in idx]
      features += list(feature)
    assert len(indices) == len(features)
    return indices, features


class DependencyVectorizerWithPOS(DependencyVectorizer):
  pass
