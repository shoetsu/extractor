<0. train_fileを用いたword2vecの作成>
stc/script/create_wv.sh

<1. train, valid, test etc...のsv化>
clustering ディレクトリ内で
python create_sv_from_wv.py seq_dataset/train10M.t vectors/train10M.txt.size200/train10M.txt.wv 
python create_sv_from_wv.py seq_dataset/valid.t vectors/train10M.txt.size200/train10M.txt.wv 

train10M.t.w2vAve
valid.t.w2vAve  が出来る
この際、あんまり大きいサイズだと何故か2.でvectorの読み込みが失敗するので、
create_sv_from_wv.pyで --yakmo_format=Trueとすると.yakmoファイルも同時に出来る


<2. yakmoを用いたtrain_fileのクラスタリング>
まずclusteringフォルダ内で
python translate_yakmo_format.py train_file すると、yakmoのフォーマット ID dim0:value0 dim1:value1....の形式になったtrain10M.t.w2vAve.yakmoが出来る

その後、同じディレクトリで
./run_yakmo  seq_dataset/train10M.t 10 > seq_dataset/train10M.t.labels10
とするとseq_dataset内に各ツイートIDに対するクラスタ番号の.labels10ファイルと、中心のベクトル情報である.centroidsファイルが出来る。

※yakmoのクラスタリングもw2vAve化も同一ツイートIDに対して一度しか行われないことに注意。

<3. valid, test ファイルのクラスタリング>
>> これ使わなくても yakmoでtest_fileを指定すればok 
python assign_to_clusters.py seq_dataset/train10M.t.labels10.centroids seq_dataset/valid.t.w2vAve > seq_dataset/valid.t.labels10
して、trainデータでクラスタリングした結果をvalidにも適用。
 


<4. 各行ごとにassign>
>> すでにdata_utilsにinitialize_domain_labels実装済み
クラスタリング結果は id: clusterのdictionaryなので、各行ごとのデータにする。
とりあえず別スクリプト。
python assign_label_for_each_line.py seq_dataset/test.selected.t seq_dataset/test.selected.t.labels10 ../seq2seq_model/dataset/processed/test.selected.t.labels10