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

def main(args):
  assert os.path.exists(args.vector_file)
  origin_vec = np.load(args.vector_file)
  print origin_vec[0]
  print len(origin_vec)

if __name__ == "__main__":
  desc = ''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-v', '--vector_file', 
                      default='all.normalized.txt.strict.10000.4gramvec',
                      type=str, help ='')
  # parser.add_argument('label_file', help ='')
  # parser.add_argument('--n_plot', default=150, type=int, help ='')
  # parser.add_argument('--limit_size',default=50000, type=int, help ='')
  # parser.add_argument('--ltype', default='prof' , help ='')
  # parser.add_argument('--output_plot_path',default='/tmp/plot.eps', help ='')
  args = parser.parse_args()
  main(args)
