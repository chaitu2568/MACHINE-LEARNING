import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math


class GaussianNB:


    def fit(self, x, y):
        class_0=x[y==0.0]
        class_1=x[y==1.0]
        para_dict={}
        para_dict['mean_0']=np.mean(class_0, axis=0)
        para_dict['vari_0']=np.var(class_0,axis=0)
        para_dict['mean_1']=np.mean(class_1,axis=0)
        para_dict['vari_1']=np.var(class_1,axis=0)
        para_dict['yis1']=class_1.shape[0]/(class_1.shape[0]+class_0.shape[0])
        para_dict['yis0']=class_0.shape[0]/(class_1.shape[0]+class_0.shape[0])
        self.para_dict = para_dict


    def gaussValue(self, x, mean, var):
        return (math.sqrt((2*np.pi*var))) * np.exp(-(((x-mean) * (x-mean))/(2*var)))


    def probValue(self, x, mean, var):
        like = 1
        for i in range(len(x)):
            like *= self.gaussValue(x[i], mean[i], var[i])
        return like


    def predict(self, X, Y):k
        N = len(X)
        y_hat = np.zeros((N , 2))
        for i in range(len(Y)):
            likely_0 = self.para_dict['yis0'] * self.probValue(X[i], self.para_dict['mean_0'], self.para_dict['vari_0'])
            likely_1 = self.para_dict['yis1'] * self.probValue(X[i], self.para_dict['mean_1'], self.para_dict['vari_1'])
            y_hat[i] = [likely_0, likely_1]
        finaly_hat = np.argmax(y_hat, axis=1)
        return np.mean(finaly_hat==Y)


class LogisRegre:

    def __init__(self, lecoef=0.01, iterations=10000, inter=True):
        self.lecoef = lecoef
        self.iterations = iterations
        self.inter = inter


    def predict(self, x, limit=0.5):
        return self.prob(x) >= limit


    def score(self, x, y):
        return (self.predict(x) == y).sum().astype(float)/len(y)


    def sigmoidfun(self,z):
        return 1/(1+np.exp(-z))


    def addone(self,x):
        onearray = np.ones((x.shape[0],1))
        return np.concatenate((onearray,x),axis=1)


    def fittin(self, x, y):
        if self.inter:
            x = self.addone(x)
        self.weights=np.zeros(x.shape[1])
        for i in range(self.iterations):
            z = np.dot(x,self.weights)
            h = self.sigmoidfun(z)
            grad = np.dot(x.T,(h-y))/y.size
            self.weights -= self.lecoef*grad


    def prob(self,x):
        if self.inter:
            x = self.addone(x)
        return self.sigmoidfun(np.dot(x, self.weights))


def main():
    data=pd.read_csv('data_banknote_authentication.csv')
    data1=data.values
    X = data1[:, :-1]
    Y = data1[:, -1]
    nsplits = 3
    kf = KFold(nsplits)
    kf.get_n_splits(data1)
    acc = np.zeros(6)
    acc1 = np.zeros(6)
    splitSize = [0.01, 0.02, 0.05, 0.1, 0.625, 1.0]
    for train_i, test_i in kf.split(data1):
        model=LogisRegre()
        model1=GaussianNB()
        for s in range(len(splitSize)):
            X_train = X[train_i]
            Y_train = Y[train_i]
            X_train0 = X_train[Y_train==0]
            Y_train0 = Y_train[Y_train==0]
            X_train1 = X_train[Y_train==1]
            Y_train1 = Y_train[Y_train==1]
            totalSize = len(X_train)
            checkSize = min(len(X_train0), len(X_train1))
            for i in range(5):
                st = np.random.choice(checkSize, int(math.ceil((splitSize[s] * totalSize)/2)))
                X_tra = np.concatenate((X_train0[st], X_train1[st]))
                Y_tra = np.concatenate((Y_train0[st], Y_train1[st]))
                model.fittin(X_tra, Y_tra)
                eff=model.score(X[test_i], Y[test_i])
                acc[s] += eff
                model1.fit(X_tra, Y_tra)
                eff=model1.predict(X[test_i], Y[test_i])
                acc1[s] += eff


    Accuracy =[]
    Accuracy1=[]

    for i in range(6):
        acc[i] /= 15
        Accuracy.append(acc[i])
        acc1[i] /= 15
        Accuracy1.append(acc1[i])

    print(Accuracy)
    print(Accuracy1)


    plt.xlabel('Fraction Size of Training Set')
    plt.ylabel('Accuracy')
    plt.scatter(splitSize, acc)
    plt.scatter(splitSize, acc1)
    plt.plot(splitSize, acc, label="Logistic regression")
    plt.plot(splitSize, acc1, label="Gaussian classifier")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
