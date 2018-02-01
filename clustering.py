# coding: utf-8
import re, argparse, os, commands, sys, collections, random
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import KMeans, DBSCAN
from utils import common
from tokenize_and_normalize import convert_num, tokenize_and_pos_tagging
try:
  import cPickle as pickle
except:
  import pickle


from nltk.tag.stanford import StanfordPOSTagger
TAGGER_DIR = '/home/shoetsu/downloads/stanford-postagger'
tagger = StanfordPOSTagger(
  TAGGER_DIR + '/models/english-left3words-distsim.tagger',
  TAGGER_DIR + '/stanford-postagger.jar'
)

KMEANS_STR = 'kmeans'
DBSCAN_STR = 'dbscan'
MODEL_NAME = 'cluster.model'
NUM = common.NUM
NONE = '-'
stop_words = set(['.', ',', '!', '?'])
vocab_condition = lambda x : True if NUM in x and not stop_words.intersection(set(x)) else False

def output_clusters(output_dir, cluster_results, all_sents, all_features):
  def _feat2str(feat):
    """
    feat: a list of tuple (tuple of n_gram, frequency) in a sentence.
    """
    return ",  ".join([common.quote(" ".join(tup)) + ":" + "%.2f" % freq for tup, freq in feat])

  labels = cluster_results.labels_
  n_clusters = len(set(labels))
  sents_by_cluster = [[] for _ in xrange(n_clusters)]
  features_by_cluster = [[] for _ in xrange(n_clusters)]
  for i, s, f in zip(labels, all_sents, all_features):
    sents_by_cluster[i].append(s)
    features_by_cluster[i].append(f)
  n_elements = [len(x) for x in sents_by_cluster]

  if hasattr(cluster_results, 'cluster_centers_'):
    np.savetxt(output_dir + '/cluster.centroids', cluster_results.cluster_centers_)

  def _get_weighted_freqency(feats):
    scores = collections.defaultdict(int)
    for k, v in common.flatten(feats):
      scores[k] += v
    return sorted([(k, v*len(k)) for k,v in scores.items()], key=lambda x: -x[1])

  with open(output_dir + '/expressions', 'w') as f:
    expressions = [' '.join(_get_weighted_freqency(feats)[0][0]) for feats in features_by_cluster]
    sys.stdout = f
    print "\n".join(set(expressions))
    sys.stdout = sys.__stdout__

  for i, (sents, feats) in enumerate(zip(sents_by_cluster, features_by_cluster)):
    with open(output_dir + '/c%03d.elements' % i, 'w') as f:
      sys.stdout = f
      print "\n".join(sents)
      sys.stdout = sys.__stdout__

    with open(output_dir + '/c%03d.features' % i, 'w') as f:
      sys.stdout = f
      print "\n".join([_feat2str(feat) for feat in feats])
      sys.stdout = sys.__stdout__

    with open(output_dir + '/c%03d.summary' % i, 'w') as f:
      sys.stdout = f
      print _feat2str(_get_weighted_freqency(feats)[:3])
      sys.stdout = sys.__stdout__
    
  with open(output_dir + '/cluster.info', 'w') as f:
    sys.stdout = f
    print common.timestamp()
    print "Num of Elements:"
    print " ".join([str(x) for x in n_elements])
    print " "
    sys.stdout = sys.__stdout__

  with open(output_dir + '/cluster.labels', 'w') as f:
    sys.stdout = f
    for l in cluster_results.labels_:
      print l
    sys.stdout = sys.__stdout__

  sys.stderr.write("The result is output to %s.\n" % output_dir)

def get_ngram_matches(test_sentences, feature_scores):
  # feature_scores: default_dict[ngram] = score
  ngram_length = set([len(k) for k in feature_scores.keys()])
  min_n = min(ngram_length)
  max_n = max(ngram_length)
  res_spans = []
  for i, s in enumerate(test_sentences):
    if type(s) == str:
      s = s.split(' ')
    test_sent_ngrams = common.get_ngram(s, min_n, max_n, vocab_condition=vocab_condition)
    possible_expr = list(set(feature_scores.keys()).intersection(test_sent_ngrams))
    possible_expr = sorted([(e, feature_scores[e]) for e in possible_expr], key=lambda x:-x[1])
    possible_expr = [e[0] for e in possible_expr]
    spans = []
    for expr in possible_expr:
      new_spans = common.get_ngram_match(s, expr)
      # Check whether the newly acquired span doesn't overlaps with the span of higher priority
      new_spans = common.check_overlaps(spans, new_spans)
      spans.extend(new_spans)
    res_spans.append(spans)
  return res_spans

class ClusterBase(object):
  def __init__(self, args):
    self.tokenizer = word_tokenize
    clustering_algorithm = args.clustering_algorithm.lower()
    if clustering_algorithm == KMEANS_STR:
      self.model = KMeans(n_clusters=args.n_clusters, random_state=0)
      self.output_dir = os.path.join(
        args.output_dir, 
        '%dgram_kmeans_c%02d' % (args.ngram_range[1], args.n_clusters))
    elif clustering_algorithm == DBSCAN_STR:
      self.model = DBSCAN()
      self.output_dir = os.path.join(
        args.output_dir, 
        '%dgram_dbscan' % (args.ngram_range[1]))
    output_dir = self.output_dir
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    if os.path.exists(os.path.join(output_dir, MODEL_NAME)):
      self.model = pickle.load(open(os.path.join(output_dir, MODEL_NAME), 'rb'))

  def evaluate(self, tests, origins, exprs):
    assert len(tests) == len(exprs)
    res = []
    for t, o, expr in zip(tests, origins, exprs):
      idx, sent, anno = t
      _, o_sent, o_anno = o
      print '<%d>' % idx
      print 'Original Sent:\t', o_sent
      print 'Tokenized Sent:\t', ' '.join(sent)
      print 'Prediction:\t', ', '.join(['\"' + ' '.join(e) + '\"' for e in expr])
      print 'Human:\t', ', '.join(['\"' + ' '.join(a) + '\"' for a in anno])


class NGramBasedClustering(ClusterBase):
  def __init__(self, args):
    super(NGramBasedClustering, self).__init__(args)
    self.vectorizer = common.NGramVectorizer(ngram_range=args.ngram_range, 
                                             min_freq=args.min_freq)
    self.vectorizer.load_vocab(self.output_dir)

  def get_features(self, sents):
    BOW = self.vectorizer.fit_transform(sents, vocab_condition=vocab_condition)
    self.vectorizer.save_vocab(self.output_dir)
    bow_vector_path = args.input_file + '.%dgramvec' % args.ngram_range[1]
    if args.cleanup or not os.path.exists(bow_vector_path):
      sys.stderr.write('Vector file: \'%s\'\n' % bow_vector_path)
      np.savetxt(bow_vector_path, BOW)

    return BOW

  @common.timewatch()
  def train(self, args, sents):
    output_dir = self.output_dir
    sents = [self.tokenizer(l) for l in sents]

    sys.stderr.write('BOW matrix: %s \n' % str(BOW.shape))
    res = self.model.fit(BOW)
    output_clusters(output_dir, res, sents, ngram_vectorizer.vec2tokens(BOW))
    pickle.dump(self.model, open(os.path.join(output_dir, MODEL_NAME), 'wb'))

  def test(self, args, sents, top_N=3):
    output_dir = self.output_dir
    BOW = ngram_vectorizer.fit_transform(sents, vocab_condition=vocab_condition)
    predictions = self.model.predict(BOW)

    feature_scores = {}
    summaries_path = commands.getoutput('ls -d %s/c*.summary' % output_dir)
    r = re.compile("\"(.+?)\":([0-9\.]+)")
    for f in summaries_path.split():
      c_idx = int(re.search('(.+)/c([0-9]+).summary', f).group(2))
      l = open(f).readline().replace('\n', '')
      top_n_features = [(tuple(k.split(' ')), float(v)) for k, v in r.findall(l)][:top_N]
      top_n_features.append((('$', NUM), 0.1))
      feature_scores[c_idx] = collections.defaultdict()
      for k, v in top_n_features:
        feature_scores[c_idx][k] = v

    res_exprs = []
    for i, (sent, c_idx) in enumerate(zip(test_sentences, predictions)):
      spans = get_ngram_matches([sent], feature_scores[c_idx])[0]
      exprs = [sent[s[0]:s[1]+1] for s in spans]
      res_exprs.append(exprs)
    return res_exprs

# def test_frequent(args, test_sentences, output_dir, top_N=1):
#   summaries_path = commands.getoutput('ls -d %s/c*.summary' % output_dir)
#   r = re.compile("\"(.+?)\":([0-9\.]+)")
#   features = [] # [(ngram, score), ...]
#   for f in summaries_path.split():
#     c_idx = int(re.search('(.+)/c([0-9]+).summary', f).group(2))
#     l = open(f).readline().replace('\n', '')
#     top_n_features = [(k, float(v)) for k,v in r.findall(l)]
#     features.append(top_n_features[:top_N])

#   feature_scores = collections.defaultdict(int)

#   for k, v in common.flatten(features):
#     k = tuple(k.split(' '))
#     feature_scores[k] = v if v >= feature_scores[k] else feature_scores[k]

#   test_sentences = [l.replace('\n', '') for l in open(args.input_file)]
#   res_spans = apply_ngrams(test_sentences, feature_scores)
#   res_exprs = []
  
#   for i, (sent, spans) in enumerate(zip(test_sentences, res_spans)):
#     exprs = [" ".join(sent.split(' ')[s[0]:s[1]+1]) for s in spans]
#     res_exprs.append(exprs)
#     print "Sent%4d:\t%s" % (i, sent)
#     print "Expr%4d:\t%s" % (i, ', '.join(exprs))
#   return res_exprs

@common.timewatch()
def read_human_annotations(args, max_len=0):
  path = 'results/candidate_sentences/corpus/corpus.origin.test.summary'
  res = collections.defaultdict()
  lines = []
  origins = []
  for i, l in enumerate(open(path)):
    if max_len and i > max_len:
      break
    idx, sent, anno = l.replace('\n', '').split('\t')
    origins.append((idx, sent, anno))
    idx = int(idx)
    sent = sent.strip()
    anno = " | ".join(anno.strip().split('|'))
    lines.append((idx, sent, anno))
  indices = [int(x[0]) for x in lines]

  sents, sents_pos = tokenize_and_pos_tagging([l[1] if l[1] else NONE for l in lines])
  annos, annos_pos = tokenize_and_pos_tagging([l[2] if l[2] else NONE for l in lines])

  annos = [[e.strip().split(' ') if not e == NONE else [] for e in ' '.join(a).split('|')] for a in annos]
  annos_pos = [p if annos[i] else [] for i, p in enumerate(annos_pos)]

  assert len(annos) == len(annos_pos) == len(sents) == len(sents_pos)
  for i, anno in enumerate(annos):
    j = 0
    tmp = []
    for a in anno:
      tmp.append(annos_pos[i][j:j+len(a)])
      j += len(a) + 1
    annos_pos[i] = tmp

  sents, sents_pos = common.unzip([convert_num(s, p) for s, p in zip(sents, sents_pos)])
  annos, annos_pos = common.unzip([common.unzip([convert_num(a, p) for a, p in zip(anno, anno_pos)]) for anno, anno_pos in zip(annos, annos_pos)])

  res = common.unzip((indices, sents, annos))
  return res, origins

@common.timewatch()
def main(args):
  model = NGramBasedClustering(args)
  if args.mode == 'train':
    sententeces = [l.replace('\n', '') for l in open(args.input_file)]
    model.train(args)
  elif args.mode == 'test':
    tests, origins = read_human_annotations(args)
    sentences = [sent for idx, sent, anno in tests]
    res_exprs = model.test(args, test_sentences)
    model.evaluate(tests, origins, res_exprs)
  else:
    raise ValueError('args.mode must be \'train\' or \'test\'.')

if __name__ == "__main__":
  random.seed(0)
  np.random.seed(0)
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', default='train')
  parser.add_argument("-i", "--input_file", default="results/candidate_sentences/corpus/all.normalized.strict.m30.0-10000", type=str, help="")
  parser.add_argument("-a", "--clustering_algorithm", default="kmeans", type=str)
  parser.add_argument("-o", "--output_dir",
                      default="results/clustering")
  parser.add_argument("-nr", "--ngram_range", default=(2,4),
                      type=common.str2tuple, help="")
  parser.add_argument("-min", "--min_freq", default=3, type=int)
  parser.add_argument("-nc", "--n_clusters", default=200, type=int)
  parser.add_argument("-cl", "--cleanup", default=False, type=common.str2bool)
  args  = parser.parse_args()
  main(args)
  
