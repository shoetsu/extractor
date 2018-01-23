# coding: utf-8
import sys, io, os, codecs, time
import argparse
from sklearn.cluster import KMeans
sys.stdout = codecs.EncodedFile(sys.stdout, 'utf-8')

from public import config
from public import utils

def main(args):
    vector_filename = args.vector_file
    origin_text = args.origin_text
    n_clusters = int(args.n_clusters)
    random_state = int(args.random_state)
    cluster_dir = args.cluster_dir

    # 元々の文とそのベクトル表現
    print "Reading sentence vector file:\t",
    t = time.time() 
    vector = utils.read_vector(vector_filename)
    print "%f sec" % (time.time() -t)
    
    print "Executing KMeans clustering:\t",
    t = time.time() 
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(vector.values())
    print "%f sec" % (time.time() -t)

    print "Create centers of the clusters:\t",
    t = time.time() 
    centroids =  kmeans_model.cluster_centers_
    print "%f sec" % (time.time() -t)

    print "Output result files:\t",
    t = time.time() 
    if not os.path.exists(cluster_dir +"/txt"):
        os.makedirs(cluster_dir +"/txt")
    with open(cluster_dir + '/cluster.info', 'w') as f:
        f.write('vector_file: %s \n' % vector_filename)
        f.write('origin_text: %s \n' % origin_text)
        f.write('n_clusters: %d \n' % n_clusters)
        f.write('random_state: %d \n' % random_state)
        f.write('N-Vector: %d \n' % len(vector))
        #f.write('Elapsed Time: %f \n' % (t-s))
    
    with open(cluster_dir + '/cluster.labels', 'w') as f:
        labels = kmeans_model.labels_
        labels_str = '\n'.join(map(lambda l: str(l),labels)) + '\n'
        f.write(labels_str)
 

    with open(cluster_dir + '/cluster.centroids', 'w') as f:
        centroids =  kmeans_model.cluster_centers_
        for c in centroids:
            f.write(' '.join(map(lambda i:str(i), c)) + '\n')

        
    if origin_text:
        sent_lines = utils.read_file(origin_text)
        for i, label in enumerate(labels):
            with open(cluster_dir + '/txt/%d.cluster' % label, 'a') as f:
                f.write('%s\n' % (utils.concatenate_sequence(sent_lines[i])))
    print "%f sec" % (time.time() -t)

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python create_clusters.py --cluster_dir=test --origin_text=test/train.onlyTW.100 10 test/train.onlyTW.100.sv')
    parser.add_argument('n_clusters', help='number of clusters')
    parser.add_argument('vector_file', help ='text used to build sentence-vectors')
    parser.add_argument('--origin_text' , default = None, help ='original texts')
    parser.add_argument('--cluster_dir', default = '',help='directory where results will be placed')
    parser.add_argument('--random_state', default='10', help='number of trials to find the best clusters')
    args = parser.parse_args()
    main(args)
    

