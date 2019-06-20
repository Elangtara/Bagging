import numpy as np 
import operator

def DatabyClass(data, classIndex) :
#Membagi date dengan index class yang diberikan
	Data ={}
	for i in range(len(data)):
		class_value = data[i][classIndex]
		if (class_value not in Data):
			Data[class_value]=[]
		Data[class_value].append(data[i][:classIndex])
	for class_ in Data:
		Data[class_]=np.array(Data[class_])
	return Data

def HitungRata(data):
#Fungsi ini akan menghitung rata-radta dan variance value untuk setiap atribut
	Rata_={}
	for class_ in data.keys():
		Rata_[class_]=[]
		for i in range(data[class_].shape[1]) :
			Rata_Atribute = np.mean(data[class_][:, i])
			Var_Atribute = np.var(data[class_][:, i]) *	( data[class_].shape[0]/(data[class_].shape[0]-1) )
			Rata_[class_].append([Rata_Atribute,Var_Atribute])
	return Rata_

def post_prob(Rata_,i_Data):
#Menghitung probabilitas Posterior dari input data dari setiap class
	phi = 3.14
	num_attribute = len(i_Data)
	p = {}
	for class_ in Rata_ :
		temp=1
		p[class_] = []
		for j in range(num_attribute):
			mean = Rata_[class_][j][0]
			var = Rata_[class_][j][1]
			temp *= (1/np.sqrt(2*phi*var)* np.exp(-0.5*(i_Data[j]-mean)**2/ var))
		p[class_]= temp
	return p

def naiveBayes(dataTrain, dataTest, classIndex):
	dataTrain_n =len(dataTrain)
	data = DatabyClass(dataTrain, classIndex)
	Rata_ = HitungRata(data)
	list_predict = []
	for row in dataTest:
		p= post_prob(Rata_,row)
		class_Prob= {}
		p ={}
		#Menghitung probabilitas setiap Class
		for class_ in data :
			class_Prob[class_]= len(data[class_]) / dataTrain_n
			p[class_]=1

		#Menghitung kemunkinan coditional setiap class
		total_prob =0.0
		for class_ in data:
			total_prob= total_prob+ (p[class_]* class_Prob[class_])
		for class_ in data :
			p[class_]=(p[class_]* class_Prob[class_])/total_prob

		predict= max(p.items(), key= operator.itemgetter(1))[0]
		list_predict.append(predict)
	return list_predict

	
