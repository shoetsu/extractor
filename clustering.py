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
NUMBER = common.NUMBER
NONE = '-'
stop_words = set(['.', ',', '!', '?'])
VOCAB_CONDITION = lambda x : True if set([NUM, NUM.lower(), NUMBER, NUMBER.lower()]).intersection(x) and not stop_words.intersection(set(x)) else False


#####################################
##        Extraction
#####################################

def spans2exprs(spans, line):
  return [tuple(line[s[0]:s[1]+1]) for s in spans]

# Depricated. (This function extracts patterns with assuming that one line belongs to only one cluster)

def get_ngram_matches(line, feature_scores):
  # feature_scores: default_dict[ngram] = score
  ngram_length = set([len(k) for k in feature_scores.keys()])
  min_n = min(ngram_length)
  max_n = max(ngram_length)
  res_spans = []
  if type(line) == str:
    line = line.split(' ')
  test_sent_ngrams = common.flatten(common.get_ngram(line, min_n, max_n, vocab_condition=VOCAB_CONDITION))
  possible_expr = list(set(feature_scores.keys()).intersection(test_sent_ngrams))
  possible_expr = sorted([(e, feature_scores[e]) for e in possible_expr], key=lambda x:-x[1])
  possible_expr = [e[0] for e in possible_expr]
  spans = []

  for expr in possible_expr:
    new_spans = common.get_ngram_match(line, expr)
    # Check whether the newly acquired span doesn't overlaps with the span of higher priority
    new_spans = [ns for ns in new_spans if common.no_overlaps(spans, ns)]
    spans.extend(new_spans)
  return spans


def extract_around_target(line, t_idx, patterns):
  res = []
  for ngram, score in patterns.items():
    min_w = max(t_idx+1-len(ngram), 0)
    max_w = min(t_idx+1, len(line) - len(ngram))
    for i in xrange(min_w, max_w):
      span = (i, i+len(ngram)-1)
      existing_spans = [s for s, _ in res]
      if tuple(line[span[0]:span[1]+1]) == ngram and common.no_overlaps(existing_spans, span):
        res.append((span, score))
  return res


########################################
##         Evalaution
########################################
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

def save_config(args):
  sys.stderr.write(str(args) + '\n')
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
  tmp_vals = ['output_dir', 'mode', 'debug', 'cleanup', 'test_file']
  restored_vals = {k:v for k, v in args.__dict__.items() if k not in tmp_vals}
  if False and os.path.exists(os.path.join(args.output_dir, CONFIG_NAME + '.bin')):
    config_path = os.path.join(args.output_dir, CONFIG_NAME + '.txt')
    msg = "Remove the old configs? [Y/n] (%s)" 
    common.ask_yn(msg, os.system, ('rm -r %s' % (config_path)))

  #pickle.dump(restored_vals, 
  #            open(os.path.join(args.output_dir, CONFIG_NAME + '.bin'), 'wb'))
  with open(os.path.join(args.output_dir, CONFIG_NAME) + '.txt', 'w') as f:
    for k,v in restored_vals.items():
      type_name = re.search("<type '(.+?)'>", str(type(v))).group(1)
      line = '%s\t%s\t%s\n' % (k,v, type_name)
      f.write(line)


def load_config(args):
  if os.path.exists(os.path.join(args.output_dir, CONFIG_NAME + '.txt')):
    config = collections.defaultdict()
    for l in open(os.path.join(args.output_dir, CONFIG_NAME + '.txt')):
      k, v, type_name = l.replace('\n', '').split('\t')
      if type_name == 'tuple':
        config[k] = common.str2tuple(v)
      elif type_name == 'int':
        config[k] = int(v)
      elif type_name == 'float':
        config[k] = float(v)
      else:
        config[k] = v
    config = common.dotDict(config)
  else:
    raise ValueError('No config file is found.')
  sys.stderr.write(str(config)+'\n')
  return config



########################################
##         Models
########################################

class ExtractBase(object):
  def __init__(self, args, config):
    self.output_dir = args.output_dir
    self.tokenizer = word_tokenize
    self.config = config

  def evaluate(self, tests, origins, predictions, N=4, cluster_ids=None):
    try:
      assert len(tests) == len(predictions)
    except:
      raise ValueError('length of (tests, predictions) = (%d, %d)' % (len(tests), len(predictions)))
    res_em = []
    res_ngm = []
    for i, (t, o, pred) in enumerate(zip(tests, origins, predictions)):
      idx, sent, gold = t
      _, o_sent, _ = o
      gold = [tuple(g) for g in gold]
      TP, gold_ngrams, pred_ngrams = ngram_matching(gold, pred, N)
      res_ngm.append((len(TP), len(gold_ngrams), len(pred_ngrams)))
      TP, FP, FN = exact_matching(gold, pred)
      res_em.append((len(TP), len(TP+FN), len(TP+FP)))

      em = 'EM_Success' if len(TP) == len(TP+FP+FN) else 'EM_Failure'
      if not cluster_ids:
        cluster_id = '*'
      elif type(cluster_ids[i]) == int:
        cluster_id = "c%03d" % cluster_ids[i]
      elif type(cluster_ids[i]) == list:
        cluster_id = " ".join(["c%03d" % c for c in cluster_ids[i]])
      else:
        raise ValueError
      print '<%d (%s)>:\t%s' % (idx, cluster_id, em)
      print 'Original Sent :\t', o_sent
      print 'Tokenized Sent:\t', ' '.join(sent)
      print 'Prediction    :\t', ', '.join(['\"' + ' '.join(e) + '\"' for e in pred])
      print 'Human         :\t', ', '.join(['\"' + ' '.join(g) + '\"' for g in gold])


    # Calculate precisions and recalls from the results of all lines together.
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


class ClusterBase(ExtractBase):
  def __init__(self, args, config):
    super(ClusterBase, self).__init__(args, config)
    self.top_N = 5 
    if self.config.clustering_algorithm == KMEANS_STR:
      self.model = KMeans(n_clusters=self.config.n_clusters, random_state=0)
    elif self.config.clustering_algorithm == DBSCAN_STR:
      self.model = DBSCAN()

    if os.path.exists(os.path.join(self.output_dir, MODEL_NAME)):
      self.model = pickle.load(open(os.path.join(self.output_dir, MODEL_NAME), 'rb'))

  def get_features(self, sents):
    raise NotImplementedError


class NGramBasedClustering(ClusterBase):
  def __init__(self, args, config):
    super(NGramBasedClustering, self).__init__(args, config)
    vectorizers = []
    for idx, feature_type in enumerate(self.config.feature_type.split(',')):
      vectorizer = getattr(utils.features, feature_type)(
        idx, self.output_dir,
        ngram_range=self.config.ngram_range, 
        min_freq=self.config.min_freq,
        vocab_size=self.config.vocab_size,
        vocab_condition=VOCAB_CONDITION)
      vectorizers.append(vectorizer)
    self.vectorizer = utils.features.MultiVectorizerWrapper(vectorizers)

  def output_training(self, indices, cluster_results, 
                      all_lines, all_features):
    output_dir = self.output_dir
    def _feat2str(feat):
      """
      feat: a list of tuple (tuple of n_gram, frequency) in a sentence.
      """
      return ",  ".join([common.quote(" ".join(tup)) + ":" + "%.2f" % freq for tup, freq in feat])

    labels = cluster_results.labels_
    n_clusters = len(set(labels))
    lines_by_cluster = [[] for _ in xrange(n_clusters)]
    features_by_cluster = [[] for _ in xrange(n_clusters)]
    indices = indices if indices else [(i, '*') for i in xrange(len(all_lines))]
    for label, (l_idx, _), f in zip(labels, indices, all_features):
      l = all_lines[l_idx]
      lines_by_cluster[label].append(l)
      features_by_cluster[label].append(f)
      n_elements = [len(x) for x in lines_by_cluster]

    if hasattr(cluster_results, 'cluster_centers_'):
      np.savetxt(output_dir + '/cluster.centroids', cluster_results.cluster_centers_)
    # Practically, this weighting didn't make large effects to the results.
    def _get_weighted_frequency(feats):
      scores = collections.defaultdict(int)
      for k, v in common.flatten(feats):
        scores[k] += v
      return sorted([(k, v*len(k)) for k,v in scores.items()], key=lambda x: -x[1])
    for i, (lines, feats) in enumerate(zip(lines_by_cluster, features_by_cluster)):
      if type(lines[0]) != str:
        lines = [' '.join(s) for s in lines]
      with open(output_dir + '/c%03d.elements' % i, 'w') as f:
        sys.stdout = f
        print "\n".join(lines)
        sys.stdout = sys.__stdout__

      with open(output_dir + '/c%03d.summary' % i, 'w') as f:
        sys.stdout = f
        print _feat2str(_get_weighted_frequency(feats)[:self.top_N])
        sys.stdout = sys.__stdout__

      with open(output_dir + '/c%03d.features' % i, 'w') as f:
        sys.stdout = f
        print "\n".join([_feat2str(feat) for feat in feats])
        sys.stdout = sys.__stdout__
    return None

  def get_patterns_with_score(self):
    summaries_path = commands.getoutput('ls -d %s/c*.summary' % self.output_dir)
    feature_scores = {}
    r = re.compile("\"(.+?)\":([0-9\.]+)")
    for f in summaries_path.split():
      c_idx = int(re.search('(.+)/c([0-9]+).summary', f).group(2))
      l = open(f).readline().replace('\n', '')
      top_n_features = [(tuple(k.split(' ')), float(v)) for k, v in r.findall(l)][:self.top_N]
      top_n_features.append((('$', NUM), 0.1))
      feature_scores[c_idx] = collections.defaultdict()
      for k, v in top_n_features:
        feature_scores[c_idx][k] = v
    return feature_scores

  def extract(self, indices, lines, cluster_ids):
    
    # When indices are provided the length of lines and indices can be different since indices (and cluster_ids) are assigned to each NUM token appearing in a line.
    patterns_with_scores = self.get_patterns_with_score()
    aligned_cluster_ids = None
    if not indices == None:
      # Align.
      idx_by_line = [[] for _ in xrange(len(lines))]
      aligned_cluster_ids = [[] for _ in xrange(len(lines))]
      for (l_idx, t_idx), c_id in zip(indices, cluster_ids):
        idx_by_line[l_idx].append((t_idx, c_id))
        aligned_cluster_ids[l_idx].append(c_id)
      predictions = []
      for line, idxs in zip(lines, idx_by_line):
        spans = common.flatten([extract_around_target(line, t_idx, patterns_with_scores[c_id]) for t_idx, c_id in idxs])
        spans = sorted(spans, key=lambda x:-x[1])
        accepted_spans = []
        #print line
        #print spans
        for new_span, score in spans:
          existing_spans = [span for span, _ in accepted_spans]
          if common.no_overlaps(existing_spans, new_span):
            accepted_spans.append((new_span, score))
        accepted_spans = sorted([span for span, _ in accepted_spans], key=lambda x:x[0])
        exprs = spans2exprs(accepted_spans, line)
        predictions.append(exprs)
    else:
      predictions = []
      for i, line in enumerate(lines):
        c_id = cluster_ids[i]
        exprs = spans2exprs(get_ngram_matches(line, patterns_with_scores[c_id]), 
                            line)
        predictions.append(exprs)
      #predictions = [spans2exprs(get_ngram_matches(line, patterns_with_scores), line) for line in lines]
    return predictions, aligned_cluster_ids

  @common.timewatch()
  def train(self):
    output_dir = self.output_dir
    sents = [l.replace('\n', '') for l in open(self.config.train_file)]
    # if os.path.exists(os.path.join(output_dir, MODEL_NAME)):
    #    msg = "Remove the old results? [Y/n] (%s)" % output_dir
    #    common.ask_yn(msg, os.system, ('rm -r %s' % output_dir))

    sents = [self.tokenizer(l) for l in sents]
    indices, features = self.vectorizer.fit(
      sents, input_filepath=self.config.train_file)
    cluster_results = self.model.fit(features)
    self.output_training(indices, cluster_results, sents, 
                         self.vectorizer.vec2tokens(features))
    pickle.dump(self.model, open(os.path.join(output_dir, MODEL_NAME), 'wb'))
    return

  @common.timewatch()
  def test(self, lines, test_filepath=None):
    indices, features = self.vectorizer.fit(lines, input_filepath=test_filepath)
    cluster_ids = self.model.predict(features)
    return self.extract(indices, lines, cluster_ids)


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
def read_human_annotations(test_filepath, max_len=0):
  res = collections.defaultdict()
  lines = []
  origins = []
  for i, l in enumerate(open(test_filepath)):
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

# ## tmp function: just to read chen's results
# def read_dplabels(): 
#   path = 'results/extraction/dplabel'
#   res = collections.OrderedDict()
#   for l in open(path):
#     l = l.replace('\n', '').split('\t')
#     if len([x.split() for x in l if x]) == 0:
#       continue
#     if l[0].isdigit():
#       idx, sent = l
#       idx = int(idx)
#       res[idx] = ""
#     else:
#       label = " | ".join(l[1].strip().split('|'))
#       res[idx] = label
#   indices = res.keys()
#   labels, labels_pos = tokenize_and_pos_tagging(res.values())
#   labels, labels_pos = split_annotations(labels, labels_pos)
#   labels, labels_pos = common.unzip([common.unzip([convert_num(a, p) for a, p in zip(label, label_pos)]) for label, label_pos in zip(labels, labels_pos)])

#   assert len(indices) == len(labels)
#   res = []
#   for idx, l in zip(indices, labels):
#     idx = int(idx)
#     if (idx > 10000 and idx <= 10500) or (idx > 10750 and idx <= 11000):
#       res.append(l)
#   return res


class NGramFrequency(ExtractBase):
  def __init__(self, args, config):
    super(self.__class__, self).__init__(args, config)
    vectorizers = []
    feature_type = self.config.feature_type.split(',')[0]
    for idx, feature_type in enumerate(self.config.feature_type.split(',')):
      self.vectorizer = getattr(utils.features, feature_type)(
        idx, self.output_dir,
        ngram_range=self.config.ngram_range, 
        min_freq=self.config.min_freq,
        vocab_condition=VOCAB_CONDITION)
    self.vocab_path = self.output_dir + '/cluster.vocab'

  def output_training(self, features):
    counts = sorted([(k, v) for k,v in collections.Counter(common.flatten(features)).items() if not self.config.min_freq or v >= self.config.min_freq], key=lambda x:-x[1])
    if self.config.vocab_size:
      counts = counts[:self.config.vocab_size]
    pickle.dump(counts, open(self.vocab_path, 'wb'))
    with open(self.vocab_path + '.txt', 'w') as f:
      for k,v in counts:
        l = '%s\t%s' % (" ".join(k), str(v))
        f.write(l)

  def train(self):
    lines = [self.tokenizer(l.replace('\n', '')) for l in open(self.config.train_file)]
    idx, features = self.vectorizer.get_features(lines, input_filepath=self.config.train_file)

    self.output_training(features)

  def test(self, lines, test_filepath=None):
    indices, features = self.vectorizer.get_features(lines, input_filepath=test_filepath)
    return self.extract(indices, lines)

  def get_patterns_with_score(self):
    #res = collections.OrderedDict()
    res = [(k, v + len(k) * 100000) for k, v in pickle.load(open(self.vocab_path, 'rb'))]
    res = collections.OrderedDict(sorted(res, key=lambda x: -x[1]))
    return res

  def extract(self, indices, lines):
    
    # When indices are provided the length of lines and indices can be different since indices (and cluster_ids) are assigned to each NUM token appearing in a line.
    patterns_with_scores = self.get_patterns_with_score()
    if not indices == None:
      # Align.
      idx_by_line = [[] for _ in xrange(len(lines))]
      for l_idx, t_idx in indices:
        idx_by_line[l_idx].append(t_idx)
      predictions = []
      for line, idxs in zip(lines, idx_by_line):
        spans = common.flatten([extract_around_target(line, t_idx, patterns_with_scores) for t_idx in idxs])
        spans = sorted(spans, key=lambda x:-x[1])
        accepted_spans = []
        for new_span, score in spans:
          existing_spans = [span for span, _ in accepted_spans]
          if common.no_overlaps(existing_spans, new_span):
            accepted_spans.append((new_span, score))
        accepted_spans = sorted([span for span, _ in accepted_spans], key=lambda x:x[0])
        exprs = spans2exprs(accepted_spans, line)
        predictions.append(exprs)
    else:
      predictions = []
      for i, line in enumerate(lines):
        exprs = spans2exprs(get_ngram_matches(line, patterns_with_scores), line)
        predictions.append(exprs)
      #predictions = [spans2exprs(get_ngram_matches(line, patterns_with_scores), line) for line in lines]
    return predictions, None

myself = sys.modules[__name__]
@common.timewatch()
def main(args):
  if args.mode == 'train':
    sys.stderr.write('Saving config...\n')
    config = common.dotDict(args.__dict__)
    save_config(args)
  else:
    sys.stderr.write('Loading config...\n')
    config = load_config(args)

  model = getattr(myself, config.model_type)(args, config)

  if args.mode == 'train':
    model.train()
    
  elif args.mode == 'test':
    tests, origins = read_human_annotations(args.test_file)
    lines = [line for idx, line, anno in tests]
    predictions, cluster_ids = model.test(lines, 
                                          test_filepath=args.test_file)
    model.evaluate(tests, origins, predictions, cluster_ids=cluster_ids)
  elif args.mode == 'evaluate':
    tests, origins = read_human_annotations(args.test_file)
    predictions = read_dplabels()
    model.evaluate(tests, origins, predictions)
  else:
    raise ValueError('args.mode must be \'train\' or \'test\'.')




if __name__ == "__main__":
  random.seed(0)
  np.random.seed(0)
  parser = argparse.ArgumentParser()
  parser.add_argument("output_dir")
  parser.add_argument("--train_file", default='results/candidate_sentences/corpus/all.normalized.strict.m30.0-10000', type=str, help="")
  parser.add_argument("-a", "--clustering_algorithm", default="kmeans", type=str)
  parser.add_argument("-nr", "--ngram_range", default=(2,7),
                      type=common.str2tuple, help="")
  parser.add_argument("-min", "--min_freq", default=3, type=int)
  parser.add_argument("-nc", "--n_clusters", default=100, type=int)
  parser.add_argument("-f", "--feature_type", default='NGramVectorizer', type=str)
  #parser.add_argument("-f", "--feature_type", default='DependencyVectorizer', type=str)
  parser.add_argument("-mt", "--model_type", default="NGramFrequency", help='[NGramFrequency|NGramBasedClustering]')
  parser.add_argument("-v", "--vocab_size", default=0, type=int)

  # Variables not restored
  parser.add_argument('-m', '--mode', default='train')
  parser.add_argument("--test_file", default='results/candidate_sentences/corpus/corpus.origin.test.summary', type=str, help="")
  parser.add_argument("-cl", "--cleanup", default=False, type=common.str2bool)
  parser.add_argument("-d", "--debug", default=False, type=common.str2bool)
  args  = parser.parse_args()
  main(args)
  
