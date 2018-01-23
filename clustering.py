# coding: utf-8
import re, argparse, os, commands, sys, collections
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import KMeans, DBSCAN
from utils import common

NUM = common.NUM


def output_clusters(output_dir, cluster_results, all_sents, all_features):
  def _feat2str(feat):
    return ",  ".join([common.quote(" ".join(tup)) + ":" + "%.2f" % freq for tup, freq in feat])

  labels = cluster_results.labels_
  n_clusters = len(set(labels))
  sents_by_cluster = [[] for _ in xrange(n_clusters)]
  features_by_cluster = [[] for _ in xrange(n_clusters)]
  for i, s, f in zip(labels, all_sents, all_features):
    sents_by_cluster[i].append(s)
    features_by_cluster[i].append(_feat2str(f))
  n_elements = [len(x) for x in sents_by_cluster]

  # Output
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

    
  for i, (sents, feats) in enumerate(zip(sents_by_cluster, features_by_cluster)):
    with open(output_dir + '/c%02d.elements' % i, 'w') as f:
      sys.stdout = f
      print "\n".join(sents)
      sys.stdout = sys.__stdout__

    with open(output_dir + '/c%02d.features' % i, 'w') as f:
      sys.stdout = f
      print "\n".join(feats)
      sys.stdout = sys.__stdout__

  with open(output_dir + '/cluster.info', 'w') as f:
    sys.stdout = f
    print common.timestamp()
    print "Num of Elements:"
    print " ".join([str(x) for x in n_elements])
    print " "
    # if 'cluster_centers_' in dir(cluster_results):
    #   centroids = cluster_results.cluster_centers_
    #   print "Centroids:"
    #   print "\n".join([str(" ".join([str(x) for x in c])) for c in centroids])
    sys.stdout = sys.__stdout__
  sys.stderr.write("The result is output to %s.\n" % output_dir)
    

@common.timewatch()
def main(args):
  ############################################
  tokenizer = lambda x: x
  sents = [tokenizer(l.replace('\n', '')) for l in open(args.input_file)]
  stop_words = set(['.', ',', '!', '?'])
  vocab_condition = lambda x : True if NUM in x and not stop_words.intersection(set(x)) else False
  ngram_vectorizer = common.NGramVectorizer(ngram_range=args.ngram_range, 
                                            min_freq=args.min_freq)
  BOW = ngram_vectorizer.fit_transform(sents, vocab_condition=vocab_condition)
  sys.stderr.write('BOW matrix: %s \n' % str(BOW.shape))
  # BOW = CountVectorizer(ngram_range=args.ngram_range,
  #                       stop_words=None).fit_transform(sents)

  clustering_algorithm = args.clustering_algorithm.lower()
  timewatch = common.timewatch()
  kmeans_f = timewatch(KMeans(n_clusters=args.n_clusters, random_state=0).fit)
  dbscan_f = timewatch(DBSCAN().fit)
  if clustering_algorithm == 'kmeans':
    output_dir = os.path.join(
      args.output_dir, 
      '%dgram_kmeans_c%02d' % (args.ngram_range[1], args.n_clusters))
    res = kmeans_f(BOW, kmeans_f)
    output_clusters(output_dir, res, sents, ngram_vectorizer.vec2tokens(BOW))
  elif clustering_algorithm == 'dbscan':
    output_dir = os.path.join(
      args.output_dir, 
      '%dgram_dbscan' % (args.ngram_range[1]))
    res = dbscan_f(BOW, dbscan_f)
    output_clusters(output_dir, res, sents, ngram_vectorizer.vec2tokens(BOW))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", 
                      default="results/candidate_sentences/all.normalized.txt.strict.10000",
                      type=str, help="")
  parser.add_argument("-a", "--clustering_algorithm", default="kmeans", type=str)
  parser.add_argument("-o", "--output_dir",
                      default="results/clustering")
  parser.add_argument("-nr", "--ngram_range", default=(1,4),
                      type=common.str2tuple, help="")
  parser.add_argument("-min", "--min_freq", default=5, type=int)
  parser.add_argument("-nc", "--n_clusters", default=10, type=int)
  args  = parser.parse_args()
  main(args)
  
