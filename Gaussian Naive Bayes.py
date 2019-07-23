import numpy as np
import pandas as pd
from sklearn.model_selection import KFold



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


    def predict(self, X, Y):
        N = len(X)
        y_hat = np.zeros((N , 2))
        for i in range(len(Y)):
            likely_0 = self.para_dict['yis0'] * self.probValue(X[i], self.para_dict['mean_0'], self.para_dict['vari_0'])
            likely_1 = self.para_dict['yis1'] * self.probValue(X[i], self.para_dict['mean_1'], self.para_dict['vari_1'])
            y_hat[i] = [likely_0, likely_1]
        finaly_hat = np.argmax(y_hat, axis=1)
        return np.mean(finaly_hat==Y)


data=pd.read_csv('data_banknote_authentication.csv')
data.columns=['var','skew','curt','entropy','classl']
data1=data.values
X = data1[:, :-1]
Y = data1[:, -1]
nsplits=3
kf=KFold(nsplits)
kf.get_n_splits(data1)
predictedlabel=[]

checked = []
new_points = np.zeros((400, 4))
for train_i, test_i in kf.split(data1):
    model=GaussianNB()
    model.fit(X[train_i], Y[train_i])
    X_1 = X[Y==1]
    for x in X_1:
        checked.append(tuple(x))
    mean_new = model.para_dict['mean_1']
    var_new = model.para_dict['vari_1']
    stadev = np.sqrt(var_new)
    new_sample=np.random.normal(mean_new,stadev,(1000,4))
    i = 0
    s = 0
    t = 0
    for i in new_sample:
        r = tuple(i)
        t = t + 2
        if r not in checked:
            new_points[s] = np.asarray(i)
            s += 1
        if s == 400:
            break
    gen_mean=np.mean(new_sample,axis=0)
    gen_var=np.var(new_sample,axis=0)
    print("Means of original test data and generated data are respectively")
    print(mean_new)
    print(gen_mean)

    print("Variance of original test data and generated data are respectively")
    print(gen_var)
    print(var_new)
    mean_deviated=np.abs(gen_mean-mean_new)
    var_deviateed=np.abs(gen_var-var_new)
    print("Difference of means are ")
    print(mean_deviated)
    print("Difference of Variances are ")
    print(var_deviateed)























