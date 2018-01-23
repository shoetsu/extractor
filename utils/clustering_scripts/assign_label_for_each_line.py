#coding:utf-8
import argparse, os

from public import utils

# クラスタリングされたラベルはIDをキーとするdictなので、重複ID含む各行に割当
def main(args):
  source_dir, text_file = utils.separate_path_and_filename(args.text_file)
  source_dir, label_file = utils.separate_path_and_filename(args.label_file)

  text_path = os.path.join(source_dir, text_file)
  label_path = os.path.join(source_dir, label_file)

  id_label_dict = {}
  with open(label_path, 'r') as f:
    for l in f:
      l = l.split()
      id_label_dict[l[0]] = l[1]

  with open(text_path, 'r') as f:
    ids = [l.split()[0] for l in f]
  
  for _id in ids:
    print id_label_dict[_id]
    


if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('text_file', help ='')
  parser.add_argument('label_file', help ='')
  args = parser.parse_args()
  main(args)
