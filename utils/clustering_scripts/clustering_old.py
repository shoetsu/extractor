#coding: utf-8
import sys, io, os, codecs, time, itertools, math, random, argparse, re
from logging import FileHandler
import numpy as np
from sklearn.cluster import KMeans
import collections
from logging import FileHandler

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(SCRIPT_DIR + "/public")

# my sources
import config
import utils


def output_cluster_texts(source_file, cluster_path, t_ids, t_texts, 
                         do_output=False):
    ids, labels =  utils.read_stc_file(cluster_path + '/cluster.labels')
    labels = [int(i[0]) for i in labels]
    labels_for_sv_ids = dict(zip(ids, labels))
    #labels_for_sv_ids = utils.read_stc_file(cluster_path + '/cluster.labels')
    centroids = utils.read_file(cluster_path + '/cluster.centroids', float)
    
    n_elements = [len(filter(lambda x:x==i, labels)) for i in xrange(len(centroids))]
    #labels_for_sv_ids = dict(zip(sv.keys(), labels))
    print 'n_centroids: ', len(centroids)
    print 'n_elements: ', len(labels_for_sv_ids)
    print 'n_elements_per_cluster: ', n_elements

    if do_output:
        clustered_text_path = cluster_path + '/texts'
        t_idxs = [] # 各クラスタのツイートの元テキスト上での位置
        if not os.path.exists(clustered_text_path):
            os.mkdir(clustered_text_path)
        for c_idx in xrange(len(centroids)):
            t_idxs.append([i for i, t_id in enumerate(t_ids) if labels_for_sv_ids[t_id] == c_idx])

        for c_idx in xrange(len(centroids)):
            p = '%s/%s.c%d' % (clustered_text_path, source_file, c_idx) 
            with open(p ,'w') as f:
                for t_idx in t_idxs[c_idx]:
                    text = "%s\t%s\n" % (t_ids[t_idx], ' '.join(t_texts[t_idx]))
                    f.write(text)

def run_kmeans(n_clusters, cluster_path, sv, random_state=10):
    if os.path.exists(cluster_path + '/cluster.labels'):
        _ids, _labels = utils.read_stc_file(cluster_path+'/cluster.labels')
        _labels = [int(i[0]) for i in _labels]
        centroids = utils.read_file(cluster_path + '/cluster.centroids', float)
        labels_for_sv_ids = dict(zip(_ids, _labels))
        return labels_for_sv_ids, centroids
    
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(sv.values())
    centroids =  kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    labels_for_sv_ids = dict(zip(sv.keys(), labels))

    with open(cluster_path + '/cluster.labels', 'w') as f:
        ids = sv.keys()
        labels = kmeans_model.labels_
        for i, l in enumerate(labels):
            labels_str = '%s\t%d\n' % (ids[i], l)
            f.write(labels_str)

    with open(cluster_path + '/cluster.centroids', 'w') as f:
        for c in centroids:
            f.write(' '.join(map(lambda i:str(i), c)) + '\n')
    return labels_for_sv_ids, centroids

def create_sv_by_average_wv(sentences, wv, 
                            ids=None, sv_path=None, logger=None, 
                            normalize=None):
    if normalize == 'word':
        s2v_ext = '.w2vAve.norm_w'
    elif normalize == 'sent':
        s2v_ext = '.w2vAve.norm_s'
    else:
        s2v_ext = '.w2vAve'
    sv_path = sv_path + sv2_ext 

    if sv_path and os.path.exists(sv_path):
        sv = utils.read_vector(sv_path)
        return sv

    # svはツイートID毎に作られるので、重複したツイートに対しても1つ
    sv = collections.OrderedDict({})
    wv_dim = len(wv[wv.keys()[0]])
    for i, sentence in enumerate(sentences):
        sent_id = ids[i] if ids else 'sent_' + str(i)
        vecs = [wv[word] for word in sentence if word in wv]
        vec = np.average(vecs, axis=0) if len(vecs) > 0 else np.zeros(wv_dim)
        if normalize == 'sent':
            vec = normalize(vec)
        sv[sent_id] = vec
    if sv_path:
        with open(sv_path, 'w') as f:
            f.write('%d %d\n' % (len(sv), len(sv[sv.keys()[0]]))) 
            for sent_id, vec in sv.iteritems():
                f.write('%s %s\n' % (sent_id, ' '.join(map(lambda x: str(x), vec))))

    return sv

def distance(a, b):
    return np.linalg.norm(a-b)

def normalize(raw):
    norm = np.linalg.norm(raw)
    normalized = [float(i)/norm for i in raw]
    return normalized

def find_nearest_n_sv_to_center(sv, labels_for_sv_ids, centroids, n):
    res = []
    for c_id, centroid in enumerate(centroids):
        sv_ids = [_id for _id, _label in labels_for_sv_ids.items() if c_id == _label]
        distances = [(sv_id, distance(sv[sv_id], centroid)) for sv_id in sv_ids]
        distances = sorted(distances, key=lambda x: x[1])[:n]
        res.append([d[0] for d in distances])
    return res







def create_clusters():
    ###########################
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', help='')
    parser.add_argument('n_clusters',type=int, help='number of clusters')
    parser.add_argument('--cluster_path', default='dataset/clusters',help='directory where results will be placed')
    
    parser.add_argument('--wv_path', default='dataset/vectors/tkl201301train.wv',help ='text used to build sentence-vectors')
    #dddparser.add_argument('--wv_path', default='dataset/vectors/tmp',help ='text used to build sentence-vectors')
    parser.add_argument('--random_state', type=int, default=10, help='number of trials to find the best clusters')
    parser.add_argument('--normalize', help='None | word | sent')
    
    args = parser.parse_args()
    ###########################

    n_clusters = args.n_clusters
    source = args.source_path
    wv_path = args.wv_path

    #r = re.sub('\.t$', ".r", t)
    _, source_file = utils.separate_path_and_filename(args.source_path)
    cluster_path = args.cluster_path + '/' + source_file + '.cluster%d' % n_clusters
    sv_path = cluster_path + '/' + source_file
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)
    logger = utils.logManager(handler=FileHandler(cluster_path + '/cluster.log'))
    logger.info(args)

    s = time.time()
    t_ids, t_texts = utils.read_stc_file(source)
    logger.info('READ DATASET : %f' % (time.time() - s))
    
    s = time.time()
    wv = utils.read_vector(wv_path)
    if args.normalize == 'word':
        for k in wv.keys():
            wv[k] = normalize(wv[k])
    logger.info('READ WV: %f' % (time.time() - s))


    sv = create_sv_by_average_wv(t_texts, wv, 
                                 ids=t_ids, sv_path=sv_path, logger=logger)
    logger.info('CREATE SV : %f' % (time.time() - s))

    s = time.time()
    labels_for_sv_ids, centroids = run_kmeans(n_clusters, cluster_path, sv)
    logger.info('CREATE CLUSTERS : %f' % (time.time() - s))

    s = time.time()
    print cluster_path
    output_cluster_texts(source_file, cluster_path, t_ids, t_texts, do_output=True)
    logger.info('OUTPUT CLUSTER TEXTS : %f' % (time.time() - s))

 

def visualize_by_tsne():
    import tsne
    parser = argparse.ArgumentParser()
    parser.add_argument('vector_path', help='')
    parser.add_argument('label_path', help='')
    args = parser.parse_args()
    ids, tsne_vec = tsne.calc(args.vector_path)
    cluster_ids = [int(l[1]) for l in utils.read_file(args.label_path, delimiter='\t')]
    tsne.show_colored_scatter(tsne_vec, cluster_ids, args.vector_path)

if __name__ == "__main__":
    pass
    create_clusters()
    #visualize_by_tsne()
    
