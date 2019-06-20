import numpy as np 
import pandas as pd 
import collections
import operator
import Naive as nb

np.random.seed(1)
Trainset = pd.read_csv('TrainsetTugas4ML.csv').values
Testset = pd.read_csv('TestsetTugas4ML.csv').values
new_testset = []

for i in range (len(Testset)) :
	new_testset.append(Testset[i][:2])
Testset = np.array(new_testset)

num_boots = 5
list_boots ={}
len_boots = 100

# membuat bootstrap dengan data random dari data train 
for i in range (num_boots) :
 	bootstrap = []
 	for j in range(len_boots) :
 		randomData= Trainset[np.random.randint(1,len(Trainset))]
 		bootstrap.append(randomData)
 	list_boots[i] = np.copy(bootstrap)

 #memprediksi hasil class dari data test dengan tiap bootstrap
Result ={}
for i in range(num_boots):
	Trainset= list_boots[i]
	Result[i]= nb.naiveBayes(Trainset,Testset, classIndex=2)


csv =Result[i]
np.savetxt("TebakanTugas4ML.csv", csv, fmt="%s")
