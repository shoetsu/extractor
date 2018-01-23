# -*- coding: utf-8 -*-
import sys,os,time
import argparse
import numpy as np
import math

from public import utils

def distance(a, b):
    return np.linalg.norm(a-b)
"""
    if len(a) != len(b):
        print 'dimensions of the two vectors must be same'
        exit(1)
    dist = 0.0
    for i in xrange(0, len(a)):
        dist += (a[i] - b[i]) * (a[i] - b[i])
    return np.sqrt(dist)
"""

def read_yakmo_centroids(centroids_file):
  with open(centroids_file, 'r') as f:
    # 初めの3行は飛ばす
    f.readline()
    f.readline()
    f.readline()
    centroids = []
    for l in f:
      l = l.split()[1:] # クラスタidを除外
      c = [float(i.split(':')[1]) for i in l] # feature_id:valueの列で並んでいる
      centroids.append(c)
    return centroids

  #[map(lambda i: float(i), row) for row in utils.read_file(args.centroids_file)]

def main(args):
  _, vector_filename = utils.separate_path_and_filename(args.vector_file)
  vectors = utils.read_vector(args.vector_file)
  centroids = read_yakmo_centroids(args.centroids_file)
  n_clusters = len(centroids)

  s = time.time()
  #各クラスタ中心との距離を元にそれぞれの文にクラスタを割当て
  for i, (key, val) in enumerate(vectors.items()):
    distances = [distance(centroid, val) for centroid in centroids]
    label = distances.index(min(distances))
    print "%s %s" % (str(key), str(label))
    
  

if __name__ == "__main__":
  desc = 'this script assigns each (text)vector to clusters created by "create_clusters.py using their centroids'
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('centroids_file', help ='')
  parser.add_argument('vector_file',help ='')
  args = parser.parse_args()
  main(args)
