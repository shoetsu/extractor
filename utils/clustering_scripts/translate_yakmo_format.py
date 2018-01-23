# coding: utf-8
import sys, io, os, codecs, time
import argparse

sys.stdout = codecs.EncodedFile(sys.stdout, 'utf-8')

#my sources
from public import utils


def to_wv2_format(vec_file):
    pass

def to_yakmo_format(vec_file):
    t = time.time()
    source = utils.read_vector(vec_file)
    print 'read_vector %f sec' % (time.time() - t)
    with open(vec_file + ".yakmo", 'w') as f:
        for key in source:
            line = "%s " % key
            line += " ".join(["%d:%f" % (i, val)for i, val 
                              in enumerate(source[key])])
            line += '\n'
            f.write(line)


def main(args):
    vec_file = args.vec_file
    to_yakmo_format(vec_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='output = [source_path].yakmo')
    parser.add_argument('vec_file', help='source file')
    #parser.add_argument('translate_type', help='yakmo | w2v')
    args = parser.parse_args()
    main(args)
    
