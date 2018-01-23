#coding: utf-8
import argparse, collections, re, commands, os, sys
import numpy as np
from public import utils


def calc_distance_from_centroid(ids, centroid, vectors, do_sort=True):
  res = []
  for _id in ids:
    v = vectors[_id]
    dist = utils.euclidean_distance(v, centroid)
    res.append((_id, dist))
  if do_sort:
    res = sorted(res, key=lambda x: x[1])
  return res

def find_around_centroids(vectors, centroid):
  distances = []
  for k, v in vectors.items():
    d = utils.euclidean_distance(v, centroid)
    distances.append((k, d))
  distances = sorted(distances, key=lambda x: x[1])
  return distances

def watch_texts(text_file, ids):
  if not ids:
    return
  command = 'cat %s | grep ' % text_file
  command += " ".join(["-e %s" % _id for _id in ids])
  os.system(command)
  #res = commands.getoutput(command)

@utils.timewatch()
def main(args):
  read_labels = utils.timewatch()(utils.read_labels)
  read_sparse_vector = utils.timewatch()(utils.read_sparse_vector)
  read_vector = utils.timewatch()(utils.read_vector)
  read_stc_file = utils.timewatch()(utils.read_stc_file)

  centroids = read_sparse_vector(args.centroid_file, header_size=3, limit_size=args.limit_size)
  labels = read_labels(args.cluster_file, type_f=int, limit_size=args.limit_size)
  vectors = read_vector(args.vector_file, limit_size=args.limit_size, vector_dict={})
  text_file = re.search("(.+)\.[tr]", args.cluster_file).group(0)

  n_clusters = len(centroids)

  ids_by_label = [[] for _ in xrange(n_clusters)]
  for _id, label in labels.items():
    ids_by_label[label].append(_id)


  neighbors = []
  for i, (cname, centroid) in enumerate(centroids.items()):
    ids = ids_by_label[i]
    distances = calc_distance_from_centroid(ids, centroid, vectors)
    ave_d = sum([x[1] for x in distances]) / len(distances) if len(distances) > 0 else 0
    cnorm = np.linalg.norm(np.array(centroid))
    print "Cluster:%s, N:%d, dist:%.2f centroid norm %.2f" % (cname, len(ids), ave_d, cnorm)
    sys.stdout.flush()
    neighbor = [_id for (_id, _) in distances[:args.n_output]]
    neighbors.append(neighbor)

  #for i, (cname, centroid) in enumerate(centroids.items()):
  #  print "Cluster:%s" % cname
  #  sys.stdout.flush()
  #  watch_texts(text_file, neighbors[i])



if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('vector_file', help ='')
  parser.add_argument('cluster_file', help ='')
  parser.add_argument('centroid_file', help ='')
  parser.add_argument('--text_file', default="")
  parser.add_argument('--limit_size', default=50000,type=int)
  parser.add_argument('--n_output', default=10,type=int)
  args = parser.parse_args()
  main(args)
