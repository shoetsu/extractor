#coding:utf-8
import re, argparse
NUM = "__NUM__"
def main(args):
  sents = [l for l in open(args.input_file)]
  print len(sents)
  sents = set([re.sub("[0-9.,]*[0-9]", NUM, l).replace("\n", "") for l in sents])
  print len(sents)
  #print "\n".join(sents)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", default="00.warc.gz.txt.extracted",
                      type=str, help="")
  args  = parser.parse_args()
  main(args)
