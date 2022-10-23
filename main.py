from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np
import tqdm
import matplotlib.pyplot as plt
### multi classification with decision tree

class Binary_inquiry_train:
    def __init__(self,X,Y): # input unnomarlized X and divided label into 0 and 1
        self.X = self.normalize(X)
        self.Y = Y
        self.W = np.random.rand(self.X.shape[1], 1)
        self.bias = np.random.rand()
        self.learning_rate = 0.01
        self.hypothesis = self.hypothesis_(self.X, self.W, self.bias)



        for i in tqdm.tqdm(range(8000)):
            self.W, self.bias, self.acc = self.GD_optimizer(self.X, self.Y, self.W, self.hypothesis, self.bias, self.learning_rate)
            self.hypothesis = self.sigmoid(np.dot(self.X, self.W) + self.bias).reshape((self.X.shape[0],))
            self.H = self.Binary_cross_entropy(self.hypothesis, self.Y)


        print("H = {}, train_acc = {}".format(self.H, self.acc))
        self.output = (self.W, self.bias)


    def normalize(self,X):
        mean = np.mean(X, axis=0, dtype=np.float32)
        std = np.std(X, axis=0, dtype=np.float32)
        X = (X - mean) / std
        return X

    def sigmoid(self,X):
        return 1/(1 + np.exp(-X))

    def Round_H_Round_W(self,X, hypothesis, Y):
        return ( np.dot(X.T,( hypothesis - Y )) / X.shape[0] ).reshape(4,1)

    def Round_H_Round_bias(self,X, hypothesis, Y):
        return np.sum( hypothesis - Y ) / X.shape[0]

    def Binary_cross_entropy(self,hypothesis, Y):
        H = - np.sum(( ( hypothesis - Y ) * np.log(hypothesis) + ( 1 - Y ) * np.log(1 - hypothesis)  ))
        return H

    def hypothesis_(self,X,W,bias):
        return self.sigmoid(np.dot(X, W) + bias).reshape((X.shape[0],))

    def SGD_optimizer(self,X,Y,W,hypothesis,bias,alpha,batch_size):
        #making...
        sampled_X,sampled_Y = shuffle(X,Y)
        sampled_X = sampled_X[0:batch_size]
        sampled_Y = sampled_Y[0:batch_size]
        W = W - alpha * self.Round_H_Round_W(sampled_X, hypothesis, sampled_Y)
        bias = bias - alpha * self.Round_H_Round_bias(sampled_X, hypothesis, sampled_Y)
        acc = self.accuracy(hypothesis, sampled_Y)
        return W, bias, acc


    def GD_optimizer(self,X,Y,W,hypothesis,bias, alpha):
        W = W - alpha * self.Round_H_Round_W(X, hypothesis, Y)
        bias = bias - alpha * self.Round_H_Round_bias(X, hypothesis, Y)
        acc = self.accuracy(hypothesis, Y)
        return W, bias, acc

    def accuracy(self,hypothesis,Y):
        predict = np.round(hypothesis)
        correct = [1 if predict[i] == Y[i] else 0 for i in range(len(Y))]
        acc = np.sum(correct) / len(correct)
        return acc

def divide_data_into_0_1(combined_data):
    temp = []
    for i in combined_data:
        if int(i[-1]) != 2:
            temp += [i]
    a =  np.array(temp)
    b = np.split(a, [-1], axis=1)
    X = b[0]
    Y = b[1].astype(np.int32)
    Y = np.squeeze(Y)  # array shape (number,1) >> (number,)

    return X,Y

def normalize(X):
    mean = np.mean(X, axis=0, dtype=np.float32)
    std = np.std(X, axis=0, dtype=np.float32)
    X = (X - mean) / std
    return X

def sigmoid(X):
    return 1/(1 + np.exp(-X))

if __name__ == '__main__':

    # data preparing
    a = load_iris()
    target = a['target']
    target_names = a['target_names']  ## setosa == 0 , versicolor == 1 , virginica == 2
    features = a['data']  ## 4 feature
    #np.random.seed(0)
    features, target = shuffle(features, target)  # fixed by np.random.seed

    feature_for_train = features[:130]
    target_for_train = target[:130].reshape(130,1)
    feature_for_test = features[130:]
    target_for_test = target[130:].reshape(150-130,1)

    #######################################
    #decision tree for specific case {0,1,2}
    #######################################

    #train for first decision {0,1} || {2}
    X = feature_for_train
    Y = np.array([0 if i == 0 or i == 1 else 1 for i in target_for_train])
    W1,bias1 = Binary_inquiry_train(X,Y).output
    print(X.shape,Y.shape)

    #train for second decision {0} || {1}
    combined_data = np.concatenate((feature_for_train, target_for_train), axis=1)
    X, Y = divide_data_into_0_1(combined_data)
    print(X.shape, Y.shape)
    W2,bias2 = Binary_inquiry_train(X,Y).output

    #test
    test_X = normalize(feature_for_test)
    test_Y = target_for_test

    acc = 0
    for i,X in enumerate(test_X):

        hypothesis = sigmoid(np.dot(X,W1) + bias1)
        predict = np.squeeze(np.round(hypothesis).astype(np.int32))

        if predict == 1:
            print("ans = {} , prediction = {}".format(test_Y[i][0], 2))
            acc += test_Y[i] == 2
        else:
            hypothesis = sigmoid(np.dot(X, W2) + bias2)
            predict = np.squeeze(np.round(hypothesis).astype(np.int32))

            if predict ==1:
                print("ans = {} , prediction = {}".format(test_Y[i][0], 1))
                acc += test_Y[i] == 1

            else:
                print("ans = {} , prediction = {}".format(test_Y[i][0], 0))
                acc += test_Y[i] == 0

    acc = acc / len(test_Y)
    print("acc = {} %".format(acc[0]*100))












