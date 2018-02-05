# coding: utf-8
import re, argparse, os, commands, sys, collections, random, copy
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.utils.linear_assignment_ import linear_assignment
from utils import common
from tokenize_and_normalize import convert_num, tokenize_and_pos_tagging
#from utils.features import NGramVectorizer, DepNGramVectorizer
import utils.features
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
CONFIG_NAME = 'config'
NUM = common.NUM
NONE = '-'
stop_words = set(['.', ',', '!', '?'])
vocab_condition = lambda x : True if set([NUM, NUM.lower()]).intersection(x) and not stop_words.intersection(set(x)) else False

def get_ngram_matches(test_sentences, feature_scores):
  # feature_scores: default_dict[ngram] = score
  ngram_length = set([len(k) for k in feature_scores.keys()])
  min_n = min(ngram_length)
  max_n = max(ngram_length)
  res_spans = []
  for i, s in enumerate(test_sentences):
    if type(s) == str:
      s = s.split(' ')
    test_sent_ngrams = common.flatten(common.get_ngram(s, min_n, max_n, vocab_condition=vocab_condition))
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

#####################################
def exact_matching(gold, pred):
  c_gold = copy.deepcopy(gold)
  c_pred = copy.deepcopy(pred)
  TP = []
  for e in pred:
    if e in c_gold:
      TP.append(e)
      c_gold.remove(e)
      c_pred.remove(e)
  FP = c_pred
  FN = c_gold
  assert len(TP+FP+FN)== len(pred + gold) - len(TP)
  return TP, FP, FN

def ngram_matching(gold, pred, N):
  # pred, gold: list of Ngrams.
  
  gold = [x for x in gold if x]
  pred = [x for x in pred if x]
  def _matching(g, p, N):
    p_ngrams = common.get_ngram(p, 1, N)
    g_ngrams = common.get_ngram(g, 1, N)
    TP = []
    FP = []
    FN = []
    for gn, pn in zip(g_ngrams, p_ngrams):
      tp, fp, fn = exact_matching(gn, pn)
      TP.extend(tp)
      FP.extend(fp)
      FN.extend(fn)
    assert len(TP + FN) == len(common.flatten(g_ngrams))
    assert len(TP + FP) == len(common.flatten(p_ngrams))
    return TP, common.flatten(g_ngrams), common.flatten(p_ngrams)#FP, FN

  # Example:
  if args.debug:
    pred = ['at $ __NUM__'.split(), "$ __NUM__ or".split()]
    gold = ["$ __NUM__ or more".split(), 'less $ __NUM__'.split(), "at least $ __NUM__".split()]
  f1 = np.zeros((len(gold), len(pred)))
  result_matrix = [[] for _ in xrange(len(gold))]
  for i in xrange(len(gold)):
    for j in xrange(len(pred)):
      tp, g_ngrams, p_ngrams = _matching(gold[i], pred[j], N)
      result_matrix[i].append(tp)
      prec = 1.0 * len(tp) / (len(g_ngrams))
      recall = 1.0 * len(tp) / (len(p_ngrams))
      f1[i][j] = 0.5 * (prec + recall)
  matching = linear_assignment(-f1)
  if args.debug:
    print pred
    print gold
    print 
    for i, j in matching:
      print gold[i], pred[j]
      print common.get_ngram(gold[i], 1, N)
      print common.get_ngram(pred[j], 1, N)
    print matching
    exit(1)
  result = []
  TP = common.flatten([result_matrix[i][j] for i, j in matching])
  gold_ngrams = common.flatten([common.flatten(common.get_ngram(g, 1, N)) for g in gold])
  pred_ngrams = common.flatten([common.flatten(common.get_ngram(p, 1, N)) for p in pred])
  return TP, gold_ngrams, pred_ngrams



class ClusterBase(object):
  def __init__(self, args):
    self.output_dir = args.output_dir
    if args.mode == 'train':
      self.config = common.dotDict(args.__dict__)
      sys.stderr.write('Saving config...\n')
      config = common.dotDict(args.__dict__)
      self.save_config(args)
    else:
      sys.stderr.write('Loading config...\n')
      self.config = config = self.load_config(args)

    self.tokenizer = word_tokenize
    clustering_algorithm = self.config.clustering_algorithm.lower()
    if clustering_algorithm == KMEANS_STR:
      self.model = KMeans(n_clusters=self.config.n_clusters, random_state=0)
    elif clustering_algorithm == DBSCAN_STR:
      self.model = DBSCAN()

    if os.path.exists(os.path.join(self.output_dir, MODEL_NAME)):
      self.model = pickle.load(open(os.path.join(self.output_dir, MODEL_NAME), 'rb'))

  def save_config(self, args):
    print args
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

    tmp_vals = ['output_dir', 'mode', 'debug', 'cleanup']
    restored_vals = {k:v for k, v in args.__dict__.items() if k not in tmp_vals}
    if os.path.exists(os.path.join(args.output_dir, CONFIG_NAME + '.bin')):
      config1 = os.path.join(args.output_dir, CONFIG_NAME + '.bin')
      config2 = os.path.join(args.output_dir, CONFIG_NAME + '.txt')
      msg = "Remove the old configs? [Y/n] (%s)" 
      common.ask_yn(msg, os.system, ('rm -r %s %s' % (config1, config2)))

    pickle.dump(restored_vals, 
                open(os.path.join(args.output_dir, CONFIG_NAME + '.bin'), 'wb'))
    with open(os.path.join(args.output_dir, CONFIG_NAME) + '.txt', 'w') as f:
      for k,v in restored_vals.items():
        print k, v, type(v)
        type_name = re.search("<type '(.+?)'>", str(type(v))).group(1)
        line = '%s\t%s\t%s\n' % (k,v, type_name)
        f.write(line)

  def load_config(self, args):
    if False and os.path.exists(os.path.join(args.output_dir, CONFIG_NAME + '.bin')):
      config = pickle.load(open(os.path.join(args.output_dir, CONFIG_NAME + '.bin'), 'rb'))
      config = common.dotDict(config)
    elif os.path.exists(os.path.join(args.output_dir, CONFIG_NAME + '.txt')):
      config = common.dotDict()
      for l in open(os.path.join(args.output_dir, CONFIG_NAME + '.txt')):
        k, v, type_name = l.replace('\n', '').split('\t')
        if type_name == 'tuple':
          config[k] = common.str2tuple(v)
        elif type_name == int:
          config[k] == int
        elif type_name == float:
          config[k] == float
        else:
          config[k] = v
    return config

  def evaluate(self, tests, origins, predictions, N=4):
    try:
      assert len(tests) == len(predictions)
    except:
      raise ValueError('length of (tests, predictions) = (%d, %d)' % (len(tests), len(predictions)))
    res_em = []
    res_ngm = []
    for i, (t, o, pred) in enumerate(zip(tests, origins, predictions)):
      idx, sent, gold = t
      _, o_sent, o_gold = o
      print '<%d>' % idx
      print 'Original Sent :\t', o_sent
      print 'Tokenized Sent:\t', ' '.join(sent)
      print 'Prediction    :\t', ', '.join(['\"' + ' '.join(e) + '\"' for e in pred])
      print 'Human         :\t', ', '.join(['\"' + ' '.join(g) + '\"' for g in gold])
      TP, gold_ngrams, pred_ngrams = ngram_matching(gold, pred, N)
      res_ngm.append((len(TP), len(gold_ngrams), len(pred_ngrams)))
      TP, FP, FN = exact_matching(gold, pred)
      res_em.append((len(TP), len(TP+FN), len(TP+FP)))

    def calc_PR(res):
      n_tp = sum([x[0] for x in res])
      n_gold = sum([x[1] for x in res])
      n_pred = sum([x[2] for x in res])
      precision = 1.0 * n_tp / n_pred
      recall = 1.0 * n_tp / n_gold
      return precision, recall

    precision, recall = calc_PR(res_em)
    print ""
    print "Precision (EM):\t%.3f " % precision
    print "Recall    (EM):\t%.3f" % recall

    precision, recall = calc_PR(res_ngm)
    print "Precision (~%d-gram):\t%.3f " % (N, precision)
    print "Recall    (~%d-gram):\t%.3f" % (N, recall)

  def get_features(self, sents):
    raise NotImplementedError


class NGramBasedClustering(ClusterBase):
  def __init__(self, args):
    super(NGramBasedClustering, self).__init__(args)
    self.vectorizer = getattr(utils.features, self.config.feature_type)(
      ngram_range=args.ngram_range, 
      min_freq=args.min_freq)
    self.vectorizer.load_vocab(self.output_dir)

  def get_features(self, sents):
    BOW = self.vectorizer.fit_transform(sents, vocab_condition=vocab_condition)
    self.vectorizer.save_vocab(self.output_dir)
    sys.stderr.write('BOW matrix: %s \n' % str(BOW.shape))
    bow_vector_path = args.train_file + '.%dgramvec' % args.ngram_range[1]
    if args.cleanup or not os.path.exists(bow_vector_path):
      sys.stderr.write('Vector file: \'%s\'\n' % bow_vector_path)
      np.savetxt(bow_vector_path, BOW)
    return BOW

  def output_clusters(self, cluster_results, all_sents, all_features):
    output_dir = self.output_dir
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

    sys.stderr.write("The result is output to %s.\n" % output_dir)
    with open(output_dir + '/expressions', 'w') as f:
      expressions = [' '.join(_get_weighted_freqency(feats)[0][0]) for feats in features_by_cluster]
      sys.stdout = f
      print "\n".join(set(expressions))
      sys.stdout = sys.__stdout__

    for i, (sents, feats) in enumerate(zip(sents_by_cluster, features_by_cluster)):
      if type(sents[0]) != str:
        sents = [' '.join(s) for s in sents]
      with open(output_dir + '/c%03d.elements' % i, 'w') as f:
        sys.stdout = f
        print "\n".join(sents)
        sys.stdout = sys.__stdout__

      with open(output_dir + '/c%03d.summary' % i, 'w') as f:
        sys.stdout = f
        print _feat2str(_get_weighted_freqency(feats)[:5])
        sys.stdout = sys.__stdout__
      # with open(output_dir + '/c%03d.features' % i, 'w') as f:
      #   sys.stdout = f
      #   print "\n".join([_feat2str(feat) for feat in feats])
      #   sys.stdout = sys.__stdout__

    # with open(output_dir + '/cluster.info', 'w') as f:
    #   sys.stdout = f
    #   print common.timestamp()
    #   print "Num of Elements:"
    #   print " ".join([str(x) for x in n_elements])
    #   print " "
    #   sys.stdout = sys.__stdout__

    # with open(output_dir + '/cluster.labels', 'w') as f:
    #   sys.stdout = f
    #   for l in cluster_results.labels_:
    #     print l
    #   sys.stdout = sys.__stdout__

  @common.timewatch()
  def train(self, args):
    output_dir = self.output_dir
    sents = [l.replace('\n', '') for l in open(args.train_file)]
    if os.path.exists(os.path.join(output_dir, MODEL_NAME)):
      msg = "Remove the old results? [Y/n] (%s)" % output_dir
      common.ask_yn(msg, os.system, ('rm -r %s' % output_dir))

    sents = [self.tokenizer(l) for l in sents]
    features = self.get_features(sents)
    res = self.model.fit(features)
    self.output_clusters(res, sents, self.vectorizer.vec2tokens(features))
    pickle.dump(self.model, open(os.path.join(output_dir, MODEL_NAME), 'wb'))
    return res

  @common.timewatch()
  def test(self, args, sents, top_N=3):
    output_dir = self.output_dir
    features = self.vectorizer.fit_transform(sents, vocab_condition=vocab_condition)
    predictions = self.model.predict(features)

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
    for i, (sent, c_idx) in enumerate(zip(sents, predictions)):
      spans = get_ngram_matches([sent], feature_scores[c_idx])[0]
      exprs = [sent[s[0]:s[1]+1] for s in spans]
      res_exprs.append(exprs)
    return res_exprs


#class DependencyBasedClustering(NGramBasedClustering):
#  def __init__(args):
    
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

#   test_sentences = [l.replace('\n', '') for l in open(args.train_file)]
#   res_spans = apply_ngrams(test_sentences, feature_scores)
#   res_exprs = []
  
#   for i, (sent, spans) in enumerate(zip(test_sentences, res_spans)):
#     exprs = [" ".join(sent.split(' ')[s[0]:s[1]+1]) for s in spans]
#     res_exprs.append(exprs)
#     print "Sent%4d:\t%s" % (i, sent)
#     print "Expr%4d:\t%s" % (i, ', '.join(exprs))
#   return res_exprs

def split_annotations(annos, annos_pos):
  annos = [[e.strip().split(' ') if not e == NONE else [] for e in ' '.join(a).split('|')] for a in annos ]
  #annos = [[e.strip().split(' ') for e in ' '.join(a).split('|')] for a in annos]

  annos_pos = [p if annos[i] else [] for i, p in enumerate(annos_pos)]
  for i, anno in enumerate(annos):
    j = 0
    tmp = []
    for a in anno:
      tmp.append(annos_pos[i][j:j+len(a)])
      j += len(a) + 1
    annos_pos[i] = tmp
  return annos, annos_pos


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
  assert len(annos) == len(annos_pos) == len(sents) == len(sents_pos)

  annos, annos_pos = split_annotations(annos, annos_pos)

  sents, sents_pos = common.unzip([convert_num(s, p) for s, p in zip(sents, sents_pos)])
  annos, annos_pos = common.unzip([common.unzip([convert_num(a, p) for a, p in zip(anno, anno_pos)]) for anno, anno_pos in zip(annos, annos_pos)])
  res = common.unzip((indices, sents, annos))
  return res, origins

## tmp function
def read_dplabels():
  path = 'results/extraction/dplabel'
  res = collections.OrderedDict()
  for l in open(path):
    l = l.replace('\n', '').split('\t')
    if len([x.split() for x in l if x]) == 0:
      continue
    if l[0].isdigit():
      idx, sent = l
      idx = int(idx)
      res[idx] = ""
    else:
      label = " | ".join(l[1].strip().split('|'))
      res[idx] = label
  indices = res.keys()
  labels, labels_pos = tokenize_and_pos_tagging(res.values())
  labels, labels_pos = split_annotations(labels, labels_pos)
  labels, labels_pos = common.unzip([common.unzip([convert_num(a, p) for a, p in zip(label, label_pos)]) for label, label_pos in zip(labels, labels_pos)])

  assert len(indices) == len(labels)
  res = []
  for idx, l in zip(indices, labels):
    idx = int(idx)
    if (idx > 10000 and idx <= 10500) or (idx > 10750 and idx <= 11000):
      res.append(l)
  return res

@common.timewatch()
def main(args):
  model = NGramBasedClustering(args)

  if args.mode == 'train':
    model.train(args)
  elif args.mode == 'test':
    tests, origins = read_human_annotations(args)
    sentences = [sent for idx, sent, anno in tests]
    res_exprs = model.test(args, sentences)
    model.evaluate(tests, origins, res_exprs)
  elif args.mode == 'evaluate':
    tests, origins = read_human_annotations(args)
    res_exprs = read_dplabels()
    model.evaluate(tests, origins, res_exprs)
  else:
    raise ValueError('args.mode must be \'train\' or \'test\'.')

if __name__ == "__main__":
  random.seed(0)
  np.random.seed(0)
  parser = argparse.ArgumentParser()
  parser.add_argument("output_dir")
  parser.add_argument('-m', '--mode', default='train')
  parser.add_argument("-i", "--train_file", default="results/candidate_sentences/corpus/all.normalized.strict.m30.0-10000", type=str, help="")
  parser.add_argument("-a", "--clustering_algorithm", default="kmeans", type=str)
  parser.add_argument("-nr", "--ngram_range", default=(2,4),
                      type=common.str2tuple, help="")
  parser.add_argument("-min", "--min_freq", default=3, type=int)
  parser.add_argument("-nc", "--n_clusters", default=100, type=int)
  parser.add_argument("-f", "--feature_type", default='DepNGramVectorizer', type=str)
  parser.add_argument("-cl", "--cleanup", default=False, type=common.str2bool)
  parser.add_argument("-d", "--debug", default=False, type=common.str2bool)
  args  = parser.parse_args()
  main(args)
  
