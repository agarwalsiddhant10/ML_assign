####### Author Details ######

#Name: Siddhant Agarwal
#Roll No.: 17CS30035

#####Execution Details#######
#Python version used: 3.5.2
#Numpy version: 1.17.0


########## Code ##############


import csv
import numpy as np 
import math
import sys
import random
np.set_printoptions(threshold=sys.maxsize)

def get_attribute_values(data, attribute):
    values=[]
    for i in range(1, data.shape[0]):
        if data[i,attribute] in values:
            pass
        else:
            values.append(data[i,attribute])
    
    return values

def preprocess_data(data):
    metadata = []
    for i in range(data.shape[1]):
        val = get_attribute_values(data, i)
        metadata.append(val)
    return metadata

def get_entropy(data):
    pos = len(np.where(data[:,-1] == 'yes')[0])
    neg = len(np.where(data[:, -1] == 'no')[0])

    p_pos = pos/(1.0 *(pos + neg))
    p_neg = neg/(1.0 *(pos + neg))
    if (p_pos == 0 or p_neg == 0):
        return 0
    entropy = -p_pos * math.log(p_pos,2) - p_neg * math.log(p_neg,2)
    return entropy


def get_gain(data, attribute, metadata):
    entropy_final = 0
    entropy_beg = get_entropy(data)
    num_val = 0
    for val in metadata[attribute]:
        data_div = data[np.where(data[:, attribute] == val)[0]]
        if data_div.shape[0] ==0:
            continue
        entropy_final += (data_div.shape[0])/(1.0 * data.shape[0])*get_entropy(data_div)
        num_val = num_val + 1
    return entropy_beg - entropy_final, num_val


def get_max_gain(data, metadata, attributes):
    max_gain = -1000000
    attribute = -1

    for i in attributes:
        gain, num_val = get_gain(data, i, metadata)
        # if num_val == 1:
        #     return -1
        if gain > max_gain:
            max_gain = gain
            attribute = i 
    
    return attribute

class Non_Leaf:
    def __init__(self, data, metadata, attributes, level, div = 0, max_level = 3):
        self.data = data
        self.metadata = metadata
        self.level = level
        self.max_level = max_level
        self.child = []
        self.div = div
        self.attributes = attributes
        # print(self.max_level)

    def set_child(self):
        self.attribute = get_max_gain(self.data, self.metadata, self.attributes)
        attributes = []
        for attr in self.attributes:
            if attr != self.attribute:
                attributes.append(attr)
        # if(self.attribute == -1):
        #     return -1
        if self.level < self.max_level :
            for val in self.metadata[self.attribute]:
                node = Non_Leaf(self.data[np.where(self.data[:, self.attribute] == val)[0]], self.metadata, attributes, self.level + 1, div = val, max_level = self.max_level)
                self.child.append(node)
            for child in self.child:
            #     val = child.div
                check = child.set_child()
            #     if check == -1:
            #         self.child.remove(child)
            #         new_child = Leaf(self.data[np.where(self.data[:, self.attribute] == val)[0]], self.metadata, div = val)
            #         new_child.get_class()
            #         self.child.append(new_child)
        else:
            for val in self.metadata[self.attribute]:
                node = Leaf(self.data[np.where(self.data[:, self.attribute] == val)[0]],  self.metadata, div = val)
                self.child.append(node)
            for child in self.child:
                child.get_class()



    def Display(self, attribute, level, arg):
        if self.div == 0:
            attr = arg[self.attribute]
            for child in self.child:
                child.Display(attr, level + 1, arg)

        if self.div !=0:
            space = ''
            for i in range(level):
                space += '  '
            print(space + '|' + attribute + ' = ' + self.div)
            attr= arg[self.attribute]
            for child in self.child:
                child.Display(attr, level + 1, arg)

    def classify(self, sample):
        val = sample[self.attribute]
        for child in self.child:
            if child.div == val:
                return child.classify(sample)


class Leaf:
    def __init__(self, data, metadata, div):
        self.data = data
        self.metadata = metadata
        self.div = div

    def get_class(self):
        pos = len(np.where(self.data[:,-1] == 'yes')[0])
        neg = len(np.where(self.data[:,-1] == 'no')[0])

        if pos > neg:
            self._class = 'yes'
        else:
            self._class = 'no'

    def Display(self, attribute, level, args):
        space = ''
        for i in range(level):
            space += '  '
        print(space + '|' + attribute + ' = ' + self.div + '   ' + self._class)

    def classify(self, sample):
        return self._class


class AdaBoost:
    def __init__(self, data, metadata, num_classifiers = 3, classifier_max_level = 1):
        self.data = data 
        self.metadata = metadata
        self.num_classifiers = num_classifiers
        self.classifier_max_level = classifier_max_level
        # print('in adaboost', self.classifier_max_level)
        self.weights = np.zeros(self.data.shape[0]-1)
        self.classifiers = []
        self.alphas = []
        self.curr_data = None


    def get_weights(self):
        if self.weights[0]==0:
            self.weights[:] = 1/(self.data.shape[0]-1)
            return

        eps = 0
        for i, sample in enumerate(self.curr_data):
            pred = self.classifiers[-1].classify(sample)
            if pred != sample[-1]:
                eps += self.weights[i]
        
        alpha = (1/2) * math.log((1-eps)/eps)
        self.alphas.append(alpha)
        for i, sample in enumerate(self.curr_data):
            pred = self.classifiers[-1].classify(sample)
            if pred != sample[-1]:
                self.weights[i] = self.weights[i]*math.exp(-alpha)

            else:
                self.weights[i] = self.weights[i]*math.exp(alpha)

        return

    def train(self):
        print('Training')
        self.get_weights()
        for iter in range(self.num_classifiers):

            self.curr_data = random.choices(population = self.data[1:], weights = self.weights, k=self.data.shape[0]-1)
            self.curr_data_t = np.vstack((self.data[0], self.curr_data))
            classifier = Non_Leaf(self.curr_data_t, self.metadata, [0, 1, 2], 1, max_level = self.classifier_max_level)
            classifier.set_child()
            self.classifiers.append(classifier)
            self.get_weights()
            print('Classifier {} trained'.format(iter + 1))
            # print(self.weights)

        
    def classify(self, sample):
        yes = 0
        no = 0
        for i in range(self.num_classifiers):
            pred = self.classifiers[i].classify(sample)
            if (pred == 'yes'):
                yes += self.alphas[i]
            else:
                no += self.alphas[i]

        if (yes > no):
            return 'yes'
        else:
            return 'no'


    def display_classifiers(self):
        for iter in range(self.num_classifiers):
            print('##############Classifier {}##############'.format(iter + 1))
            self.classifiers[iter].Display("", 1, self.data[0])
            print('########################################')
            


def main():
    f = open('data3_19.csv', 'r')
    reader = csv.reader(f)
    data = list(reader)
    data = np.array(data)
    metadata = preprocess_data(data)
    A = AdaBoost(data, metadata, num_classifiers = 3, classifier_max_level = 2)
    A.train()
    A.display_classifiers()
    count = 0
    for sample in data:
        pred = A.classify(sample)
        if pred == sample[-1]:
            count = count + 1

    print('Training Accuracy: ', count * 100.0 /data.shape[0])

    f = open('test3_19.csv', 'r')
    reader = csv.reader(f)
    data = list(reader)
    data = np.array(data)
    count = 0
    for sample in data[1:]:
        pred = A.classify(sample)
        if pred == sample[-1]:
            count = count + 1
    print('Test Accuracy: ', count * 100.0 /data.shape[0] - 1)

if __name__=='__main__':
    main()
