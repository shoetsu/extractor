import argparse
def main(args):
  for i in xrange(args.start, args.end+1):
    print "__SEP__"

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('start', type=int)
  parser.add_argument('end', type=int)
  args  = parser.parse_args()
  main(args)
