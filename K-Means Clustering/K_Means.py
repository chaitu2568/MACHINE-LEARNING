import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cosine
np.seterr(divide='ignore', invalid='ignore')

#caluculating Eucledian Distance
def eucledian(a, b):
    sum = 0
    for i in range(len(a)):
        sum += math.pow(a[i] - b[i], 2)
    return math.sqrt(sum)
#Caluclating Cosine Similarity
def cosine_similarity(a, b):
    return cosine(a, b)

class K_Means:
    def __init__(self, clu=2, con=1e-20, cost = eucledian, no_iter=10000):
        self.clu=clu
        self.no_iter=no_iter
        self.con=con
        self.sse = 0
        self.cost = cost

    def fit(self,d):
        self.centers={}
        for i in range(self.clu):
            self.centers[i]=d[np.random.choice(len(d), replace=False)] #intializing clusters centres randomly
            # self.centers[i]=d[i] # intializing clusters centres sequentially
        for i in range(self.no_iter):
            self.classes={}
            for i in range(self.clu):
                self.classes[i]=[]
            for row in d:
                space=[]
                for center in self.centers:
                    #cost is a paramter which is given to find Cosine or Eucledian
                    space.append(self.cost(self.centers[center], row))
                center_index=space.index(min(space))
                self.classes[center_index].append(row)
            old_centers=dict(self.centers)
            maxiLength = len(self.classes[0])
            maxi = self.classes[0]
            maxiPos = 0
            for poi,values in self.classes.items():
                if len(values) > maxiLength:
                    maxiLength = len(values)
                    maxi = self.classes[poi]
                    maxiPos = poi
            for i in self.classes:
                if len(self.classes[i])!=0:
                    self.centers[i]=np.mean(self.classes[i], axis=0)#calculating mean of all the points to form the new centres
                else:
                    # mid = round(len(maxi)/2)
                    # classes1=maxi[:mid]
                    # classes2=maxi[mid:]
                    # self.centers[i-1]=np.mean(classes1,axis=0)
                    # self.centers[i]=np.mean(classes2,axis=0)
                    #Dividing largest Cluster into two clusters if founds empty cluster
                    newKMeans = K_Means(clu=2)
                    newKMeans.fit(maxi)
                    self.centers[i] = newKMeans.centers[0]
                    self.centers[maxiPos] = newKMeans.centers[1]
                    # print("hi ", len(newKMeans.classes[0]), " ", len(newKMeans.classes[1]), " ", len(maxi), " ", self.clu)
            convergence = True
            for i in self.centers:
                actu_center=old_centers[i]
                pre_center=self.centers[i]
                if(np.abs(np.sum(((actu_center - pre_center)/pre_center * 100))) > self.con):
                    convergence=False
            if convergence:
                break
        self.K_index=self.classes
        self.clu_cen=self.centers
        obj_j=0
        for k in self.centers:
            for d_point in self.classes[k]:
                obj_j += np.square(eucledian(self.centers[k],d_point))
        self.sse = obj_j


def main():
    x=pd.read_csv('bc.csv',header=None)
    # print(x)
    new=x.drop([0,10],axis=1)
    # print(new)
    new_table=new.values
    sse = np.zeros(9)
    for k in range(2, 9):
        #calling K_means Implemented above
        model=K_Means(clu=k)
        model.fit(new_table)
        sse[k] = model.sse
    plt.scatter(np.arange(2, 9), sse[2:])
    plt.plot(np.arange(2, 9), sse[2:])
    plt.xlabel('Number of Clusters')
    plt.ylabel('Potential Function Value')
    plt.title('K_MEANS CLUSTERING')
    plt.show()



if __name__ == '__main__':
    main()















