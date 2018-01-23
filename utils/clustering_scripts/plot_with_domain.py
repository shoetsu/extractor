#coding:utf-8
import time, argparse

from numpy import loadtxt

#import matplotlib
#matplotlib.use('Agg') # for running at server
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#import seaborn as sns

# http://pythondatascience.plavox.info/matplotlib/%E6%95%A3%E5%B8%83%E5%9B%B3/
# http://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
# http://seesaawiki.jp/met-python/d/matplotlib
# http://qiita.com/canard0328/items/a859bffc9c9e11368f37
# http://stackoverflow.com/questions/19769262/getting-x-y-from-a-scatter-plot-with-multiple-datasets

from public import utils

class PointBrowser:
    """
    pass a subplot to __init__
    overwrite the yellow circle on the clicked data
    """
    def __init__(self, fig, ax, data, labels=None, texts=None):
        self.fig = fig
        self.data = data
        self.labels = labels
        self.texts = texts
        fname = u'/Library/Fonts/ヒラギノ角ゴ Pro W3.otf'
        fp = FontProperties(fname=fname, size=10)
        self.text = ax.text(0.05, 0.95, 'selected: none',
                            transform=ax.transAxes, va='top',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow'),
                            fontproperties=fp)


        self.selected,  = ax.plot([], [], 'o', ms=12, alpha=0.4,
                                  color='yellow', visible=False)

    def onpick(self, event):

        ind = event.ind
        x, y = self.data[ind][0]
        clicked = (x, y)

        self.selected.set_visible(True)
        self.selected.set_data(clicked)
        #text = ' x: %f\n y: %f'%(clicked)
        text = ""
        if self.labels != None:
          text += "domain: %d \n" % (int(self.labels[ind]))
        if self.texts != None:
          text += self.texts[ind]
        text = text.decode('utf-8')
        self.text.set_text(text)
        self.fig.canvas.draw()

def get_configs():
  fontsize = 14
  params = {
    'backend': 'ps', # バックエンド設定
    'axes.labelsize': fontsize, # なにが変わるのかよくわからない
    'text.fontsize': fontsize, # テキストサイズだろう　　　　　　
    'legend.fontsize': fontsize, # 凡例の文字の大きさ
    'xtick.labelsize': fontsize, # x軸の数値の文字の大きさ
    'ytick.labelsize': fontsize, # y軸の数値の文字の大きさ
    #'text.usetex': True, # 使用するフォントをtex用（Type1）に変更
    #'figure.figsize': [width, height]
  } # 出力画像のサイズ（インチ）
  return params

def main(args):
  _ , title = utils.separate_path_and_filename(args.vector_file)
  vector_file = args.vector_file
  label_file = args.label_file
  text_file = args.text_file
  img_file = vector_file + '.png' if not args.target_dir else args.target_dir + '/' + title + '.png'
  
  data = loadtxt(vector_file)
  labels = [float(l.split()[1]) for l in open(label_file)]#loadtxt(label_file) if label_file else None
  _, texts = utils.read_stc_file(text_file, tokenize_text=False) if text_file else (None, None)
  if text_file:
    texts = ["".join([w for w in l.split()]) for l in texts]

  fig = plt.figure()
  cname = 'brg'
  cmap = plt.get_cmap(cname)
  ax = fig.add_subplot(111)
  ax.set_title(title)
  browser = PointBrowser(fig, ax, data, labels, texts)
  sc = ax.scatter(data[:,0], data[:,1], #s=pointsize, 
                  cmap=cmap, picker=True, c=labels, 
                  marker='o', label='test')
  fig.canvas.mpl_connect('pick_event', browser.onpick)
  config = get_configs()
  plt.rcParams.update(config)
  if label_file:
    plt.colorbar(sc)
  plt.savefig(img_file)
  plt.show()

if __name__ == "__main__":
  desc = 'python scripts/plot_with_domain.py baseline/states/test.selected300.ep450000.h.l1.tsne --label_file=dataset/processed/test.selected300.t.domains(test.selected300.t.labels10)'
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('vector_file', help ='')
  parser.add_argument('--label_file', type=str, default='', help ='')
  parser.add_argument('--text_file', type=str, default='', help ='')
  parser.add_argument('--target_dir', type=str, default='', help ='')
  
  args = parser.parse_args()
  main(args)
