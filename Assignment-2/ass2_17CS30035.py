import numpy as np 
import csv
import pandas as pd

def pre_process(file):
    data = pd.read_csv(file)
    data = data.iloc[:, :]
    data = np.array(data)
    # print(data)
    data_final = None
    for row in data:
        # print(row)
        row = row[0].split(',')
        if data_final is None:
            data_final = np.array(row)
        else:
            data_final = np.vstack((data_final, np.array(row)))

    return data_final.astype(np.int)


def get_prob(data, col, val, class_id):
    data_ = data[np.where(data[:,0]==class_id)]
    num = data_.shape[0]
    num_pos = len(np.where(data_[:, col] == val)[0])
    return (num_pos + 1)/(num + 2)

def create_prob_mat(data):
    P = [[[get_prob(data, col, val, class_id) for val in range(1, 6)] for col in range(1, 7)] for class_id in range(2)]
    P = np.array(P)
    # print(P.shape)
    # print(P)
    return P 

def get_class_prob(data):
    num_0 = len(np.where(data[:,0]==0)[0])
    num_1 = len(np.where(data[:,0]==1)[0])
    num = data.shape[0]
    return [num_0/num , num_1/num]


def classify(P, class_p, sample):
    class_0 = P[0, 0, sample[0]-1] * P[0, 1, sample[1]-1] * P[0, 2, sample[2]-1] * P[0, 3, sample[3]-1] * P[0, 4, sample[4]-1] *P[0, 5, sample[5]-1] * class_p[0]
    class_1 = P[1, 0, sample[0]-1] * P[1, 1, sample[1]-1] * P[1, 2, sample[2]-1] * P[1, 3, sample[3]-1] * P[1, 4, sample[4]-1] *P[1, 5, sample[5]-1] * class_p[1]
    if class_0 > class_1:
        return 0
    else:
        return 1

def main():
    data = pre_process('data2_19.csv')
    # print(data)
    P = create_prob_mat(data)
    class_p = get_class_prob(data)
    print(P)
    print(class_p)
    count = 0
    for i in range(data.shape[0]):
        sample = data[i, 1:]
        class_id = data[i, 0]
        pred = classify(P, class_p, sample)
        if class_id == pred:
            count = count + 1
    print('Train Accuracy: ', count*100.0/data.shape[0])
    test = pre_process('test2_19.csv')
    count = 0
    for i in range(test.shape[0]):
        sample = test[i, 1:]
        class_id = test[i, 0]
        pred = classify(P, class_p, sample)
        if class_id == pred:
            count = count + 1
    print('Test Accuracy: ', count*100.0/test.shape[0])


if __name__=='__main__':
    main()