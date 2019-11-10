####### Author Details ######

#Name: Siddhant Agarwal
#Roll No.: 17CS30035

#####Execution Details#######

######IMPORTANT: Python 3.6 is used ##################
#Python version used: 3.6.8
######Important Please use python 3.6, random.choices is not available in python 3.5#############
#Numpy version: 1.17.2


########## Code ##############

import numpy as np 
import pandas



try:
    from tqdm import tqdm
except:
    raise Exception("Please install tqdm. Or comment out the import its use in the code")
import sys
np.set_printoptions(threshold=sys.maxsize)


if sys.version_info.major <3:
    raise Exception("Use python >= 3.6")

if sys.version_info.minor <6:
    raise Exception("Use python >= 3.6")


def get_label(data, idx):
    return data[idx, -1]

def get_features(data, idx):
    return data[idx, :-1]

class K_Means:
    def __init__(self, data, num_clusters = 3, num_iterations = 10):
        self.data = data
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        np.random.seed(0)
        mean_ids = np.random.permutation(data.shape[0])[-3:]
        self.means = data[mean_ids, :-1]
        self.mean_labels = np.unique(self.data[:,-1])
        self.processed_data = self.data
        self.processed_data = np.hstack((self.processed_data, np.zeros((self.data.shape[0], 1))))

    def L2_norm(self, feature1, feature2):
        return np.linalg.norm(feature1 - feature2, 2)

    def get_cluster_mean(self, data):
        mean = np.empty((1, 4))
        # (val, counts) = np.unique(data[:, -1] , return_counts=True)
        # mean_cl = val[np.argmax(counts)]
        mean[:,:] = np.mean(data[:,:-2], 0)
        return mean


    def set_mean(self, feature):
        return np.argmin(np.array([self.L2_norm(feature, self.means[i,:]) for i in range(self.num_clusters)]), axis = 0)


    def train(self):
        for iter in tqdm(range(self.num_iterations)):
            for i in range(self.processed_data.shape[0]):
                self.processed_data[i, -1] = self.set_mean(self.processed_data[i,0:4])
            
            for i in range(self.num_clusters):
                data_temp = self.processed_data[np.where(self.processed_data[:,-1] == i)]
                self.means[i] = self.get_cluster_mean(data_temp)

    def set_cluster_ids(self):
        # labels = []
        # ground_truth = {}
        # ground_truth[self.mean_labels[0]] = 0
        # ground_truth[self.mean_labels[1]] = 1
        # ground_truth[self.mean_labels[2]] = 2
        # self.processed_data = np.hstack((self.processed_data, np.zeros((self.processed_data.shape[0], 1))))
        # for i in range(self.processed_data.shape[0]):
        #     self.processed_data[i, -1] = ground_truth[self.processed_data[i,-3]]

        self.gt_cluster_ids = []
        self.pred_cluster_ids = []
        for cluster_name in self.mean_labels:
            self.gt_cluster_ids.append(np.where(self.processed_data[:,-2] == cluster_name))

        for i in range(self.num_clusters):
            self.pred_cluster_ids.append(np.where(self.processed_data[:,-1] == i))

    def get_jaccard_dist(self, pred_cluster_idx):
        # print(self.mean_labels.shape)
        dists = []
        for i in range(self.mean_labels.shape[0]):
            intersection = np.intersect1d(self.gt_cluster_ids[i], self.pred_cluster_ids[pred_cluster_idx])
            union = np.union1d(self.gt_cluster_ids[i], self.pred_cluster_ids[pred_cluster_idx])
            dists.append(1 - intersection.shape[0]*1.0/union.shape[0])
        dists = np.array(dists)
        min = np.min(dists)
        print('  {}'.format(pred_cluster_idx), end = "             ")
        cluster_id = self.mean_labels[np.argmin(dists)]
        for i in range(self.mean_labels.shape[0]):
            print('{0:1.2f}'.format(dists[i]), end = "            ")
        print( '--> Best cluster: {0} with jaccard distance of {1:1.2f}'.format(cluster_id,min))
        

def main():
    data = pandas.read_csv('data4_19.csv', header = None)
    data = data.iloc[:,:]
    data = np.array(data)
    K = K_Means(data, num_iterations = 10)
    K.train()
    K.set_cluster_ids()
    print()
    for i, mean in enumerate(K.means):
        print('Mean {0}: {1}'.format(i, mean))
    print()
    print('Cluster #', end = "     ")
    for cluster_name in K.mean_labels:
        print('{}'.format(cluster_name), end = "  ")
    print()
    for i in range(K.num_clusters):
        print('-----------------------------------------------------------')
        K.get_jaccard_dist(i)

    print()


if __name__ == '__main__':
    main()

