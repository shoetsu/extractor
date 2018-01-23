
# -*- coding: utf-8 -*-
import sys,os,time
import argparse
import numpy as np
import math
import re

#my sources
from public import config
from public import utils

def create_sv_by_average(ids, sentences, wv):
    vectors = {}
    wv_dim = len(wv[wv.keys()[0]])
    n_novec = 0
    for i, sentence in enumerate(sentences):
      if args.pos_filter:
        sys.stderr.write('---- %d ----' % i+ '\n')
        sys.stderr.write(' '.join(sentence) + '\n')
        sentence = utils.filter_by_pos(sentence, acceptable=['名詞', '動詞', '形容詞'])
        sys.stderr.write(' '.join(sentence) + '\n')
      vec = np.zeros(wv_dim)
      if ids:
        sent_id = ids[i]
      else:
        sent_id = 'sent_'+str(i)
      n = 0
      for w in sentence:
        if w in wv:
          vec += wv[w] 
          n += 1
      if n != 0:
        vec /= (1.0*n)
      else:
        n_novec += 1
      vectors[sent_id] = vec
    return vectors, n_novec

def create_sv_from_wv(args):
    source_path = args.source_path
    wv_file = args.wv_file
    with_id = args.with_id

    source_dir, filename = utils.separate_path_and_filename(source_path)
    output_dir = args.output_dir if args.output_dir else source_dir

    s = time.time()
    wv = utils.read_vector(wv_file)
    if args.with_id:
        ids, sentences = utils.read_stc_file(source_path)
    else:
        ids, sentences = None, utils.read_file(source_path)
    filename = output_dir + "/" + filename + '.w2vAve'
    sv, n_novec = create_sv_by_average(ids, sentences, wv)
    print "N of Documents: %d" % len(sv.keys())
    print "N of Zero Vectors: %d" % n_novec
    with open(filename, 'w') as f:
        f.write('%d %d\n' % (len(sv), len(sv[sv.keys()[0]]))) 
        for sent_id, vec in sv.iteritems():
            f.write('%s %s\n' % (sent_id, ' '.join(map(lambda x: str(x), vec))))

    with open(filename + '.yakmo', 'w') as f:
        for sent_id, vec in sv.iteritems():
            line = "%s " % sent_id
            line += " ".join(["%d:%f" % (i, val)for i, val in enumerate(vec)])
            line += '\n'
            f.write(line)


if __name__ == "__main__":
  desc =  "This script creates sentence-vectors only from their each word vector"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('source_path', help ='')
  parser.add_argument('wv_file', help ='')
  parser.add_argument('--output_dir', default=None)
  parser.add_argument('--with_id', default=True)
  parser.add_argument('--pos_filter', default=False)
  global args
  args = parser.parse_args()
  args.with_id = utils.str_to_bool(args.with_id)
  args.pos_filter = utils.str_to_bool(args.pos_filter)
  create_sv_from_wv(args)
