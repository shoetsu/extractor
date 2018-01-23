# coding: utf-8
import sys, io, os, random, re, time, argparse, commands, collections
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.manifold import TSNE
from public import utils

import sklearn.base
import bhtsne
import numpy as np


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
  def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
    self.dimensions = dimensions
    self.perplexity = perplexity
    self.theta = theta
    self.rand_seed = rand_seed

  def fit_transform(self, x):
    return bhtsne.tsne(
      x.astype(np.float64), dimensions=self.dimensions, perplexity=self.perplexity, theta=self.theta,
      rand_seed=self.rand_seed)

def tsne(x):
  TSNE = BHTSNE()
  return TSNE.fit_transform(x)


def sample(labels, V, N=300):
  sampled_labels = []
  sampled_vectors = []
  all_vecs = [[v for i, v in enumerate(V) if labels[i] == label] for label in xrange(len(set(labels)))]
  print [len(vecs) for vecs in all_vecs]
  #N = min(N, min([len(vecs) for vecs in all_vecs]))
  for label in xrange(len(set(labels))):
    vecs = all_vecs[label]
    sampled_vectors += random.sample(vecs, min(N, len(vecs)))
    sampled_labels += [label for _ in xrange(N)]
  return sampled_labels, np.array(sampled_vectors)

def get_vecs_by_label(V, labels, dinfo):
  n_clusters = len(set(labels))
  vecs_by_label = []
  for c_id in dinfo.order:
    vecs = [v for v, l in zip(V.tolist(), labels) if c_id == l]
    vecs_by_label.append(np.array(vecs))
  return vecs_by_label

def get_plot_pattern_for_domain(ltype):
  # plotするファイルで変更
  if ltype == 'prof':
    order = [6,7,9,5,0,2,8,1,3,4]
    markers = ["v", "^", "v", "^", "^", "o", "o", "o", "v", "v"] # 0-9
    legends = collections.OrderedDict({
      6: 'Animation, comics',
      7: 'Various interests',
      9: 'Occupation',
      5: 'Twitter activity',
      0: 'Music',
      2: 'Location',
      8: 'Foreign culture',
      1: 'Girls',
      3: 'English profile',
      4: 'Too short profile',
    })
    xlim = (-30, 30)
  elif ltype == 'w2v':
    order = [1,4,3,8,2,7,6,0,9,5]
    #markers = ['o','^','^','o','^','o','^','^','v','^']
    markers = ["^","^","^","^","^","^","^","^","v","^",] # 0-9
    legends = collections.OrderedDict({
      #1: 'Diverse topics & writing styles',
      1: 'Diverse topics',
      4: 'Opinions, questions',
      3: 'Food, sightseeing',
      8: 'Emotional', #, including \'!\' or \'-\'',
      2: 'Self-explosure',
      7: 'Good morning',
      6: 'Desires',
      0: 'Coming home',
      9: 'Good night',
      5: 'Reporting return',
    })
    xlim = (-30, 30)
  res = utils.dotDict({
    'markers': markers,
    'legends': legends,
    'xlim': xlim,
    'order': order
  })
  return res

def plot(ltype, labels, V, n_plot, filename=None):
  labels, V = sample(labels, V, n_plot)
  plt.figure(figsize=(10,7))
  #cname = 'brg'
  #cname='inferno'
  cname='gist_rainbow'
  cmap = plt.get_cmap(cname)
  size = 80
  fontsize = 18
  dinfo = get_plot_pattern_for_domain(ltype)
  vecs_by_label = get_vecs_by_label(V, labels, dinfo)
  #plt.xlim(dinfo.xlim)
  plt.xlim(-35, 35)
  plt.ylim(-35, 35)
  for i, vecs in enumerate(vecs_by_label):
    c_id = dinfo.order[i]
    #color = [c_id for _ in vecs]
    color = cmap(1.0*c_id/len(vecs_by_label))
    plt.scatter(vecs[:,0], vecs[:,1], linewidths=0.02, alpha=0.8, c=color,s=size,
                marker=dinfo.markers[c_id], label=dinfo.legends[c_id],
                )

  #for l, v in zip(labels, V):
  #  plt.annotate(l, xy=tuple(v),size=8)
  #plt.legend(loc='upper left', fontsize=fontsize,)
  plt.legend(fontsize=fontsize-1, numpoints=1,
             bbox_to_anchor=(0.2, 1.1), borderaxespad=0)
  plt.tick_params(labelsize=fontsize)
  #plt.subplots_adjust(left=0.7)
  #plt.subplots_adjust(wspace=0.2)
  if filename:
    if os.path.exists(filename):
      os.system('rm ' + filename)
    plt.subplots_adjust(left=0.1,)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.15)
  else:
    plt.show()
  


def load(args):
  k, l, v = [], [], []
  v_path = args.vector_file + '.tsne%d' % args.limit_size
  kl_path = args.label_file + '.tsne%d' % args.limit_size
  if os.path.exists(v_path):
    v = np.loadtxt(v_path)
  if os.path.exists(kl_path):
    for line in open(kl_path):
      line = line.split()
      k.append(line[0])
      l.append(int(line[1]))
  return k, l, v

def save(keys, labels, vectors):
  #with open(args.vector_file + '.%d.tsne' % args.limit_size, 'w'):
  np.savetxt(args.vector_file + '.tsne%d' % args.limit_size, vectors)
  with open(args.label_file + '.tsne%d' % args.limit_size, 'w') as f:
    for k, l in zip(keys, labels):
      line = '%s %d\n' % (k, l)
      f.write(line)

def to2d(args):
  vectors = utils.read_vector(args.vector_file, 
                              header_size=1, limit_size=args.limit_size)
  labels = utils.read_labels(args.label_file)
  keys = vectors.keys()
  labels = [labels[k] for k in keys]
  vectors = tsne(np.array(vectors.values()))
  save(keys, labels, vectors)
  return keys, labels, vectors

def main(args):
  print args
  k, l, v = load(args)
  if len(v) == 0:
    k, l, v = to2d(args)
  plot(args.ltype, l, v, args.n_plot, filename=args.output_plot_path)

if __name__ == "__main__":
  desc = ''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('vector_file', help ='')
  parser.add_argument('label_file', help ='')
  parser.add_argument('--n_plot', default=150, type=int, help ='')
  parser.add_argument('--limit_size',default=50000, type=int, help ='')
  parser.add_argument('--ltype', default='prof' , help ='')
  parser.add_argument('--output_plot_path',default='/tmp/plot.eps', help ='')
  args = parser.parse_args()
  main(args)
