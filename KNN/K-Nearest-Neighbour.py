import numpy as np
import pandas as pd
import operator
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class KNN():

    def __init__(self):
        self.train_x, self.train_y,self.test_x, self.test_y= self.loading_data()


    def loading_data(self):
        train=pd.read_csv('mnist_train.csv', nrows=6000)
        test=pd.read_csv('mnist_test.csv', nrows=1000)
        test_x=test.values[:,1:]
        test_y=test.values[:,0]
        train_x = train.values[:,1:]
        train_y = train.values[:,0]
        return train_x,train_y,test_x,test_y


    def euclidean(self,x,y):
        return cdist(x,y,metric='euclidean')


    def fitt(self,x,y,z,d,k=5):
        nrtrain_x=y.shape[0]
        l = (nrtrain_x,1)
        final_y = np.zeros(shape=l)
        eucli=self.euclidean(y,x)
        for point in range(nrtrain_x):
            point_dist=eucli[point]
            point_dict=dict(enumerate(point_dist))
            dist_sor = sorted(point_dict.items(),key=operator.itemgetter(1))
            dist_sor = dist_sor[0:k]
            votes=Counter()
            for index, dis in dist_sor:
                votes[z[index]] += 1
            votes_result = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
            (result, count) = votes_result[0]
            final_y[point] = result
        accuracy=np.sum(final_y==d.reshape(len(d),1))/len(d)
        print(accuracy)
        return 1-accuracy


    def run(self):
        kvalues = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        trerror=[]
        teerror=[]
        for i in kvalues:
            teerror1=[]
            trerror1=[]
            for j in range(5):
                teerror1.append(self.fitt(x=self.train_x,y=self.test_x,z=self.train_y,d=self.test_y,k=i))
                trerror1.append(self.fitt(x=self.train_x,y=self.train_x,z=self.train_y,d=self.train_y,k=i))
            teerror.append(sum(teerror1)/5)
            trerror.append(sum(trerror1)/5)
        plt.plot(kvalues,teerror,label="TestError")
        plt.plot(kvalues,trerror,label="Trainrror")
        plt.legend()
        plt.xlabel("Value of K")
        plt.ylabel("Error Rate")
        plt.savefig('KNN.png')
        plt.show()

if __name__ == "__main__":
    obj = KNN()
    obj.run()
