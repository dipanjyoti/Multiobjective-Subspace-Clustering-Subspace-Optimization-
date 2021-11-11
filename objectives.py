import numpy as np
import random
NOT_ZERO_DIVISION_SECURITY = 1e-10

def No_of_Clusters(M,member):
	Model=M
	membershipList=member
	clusters=0
	for m in range(len(membershipList)):
		if np.count_nonzero(membershipList[m]!=0):
			clusters=clusters+1
	return clusters

def dimension_non_redundancy(Model,membershipList): 
	featureSet=0
	Total_Cluster= No_of_Clusters(Model,membershipList)
	for i in range(len(Model)-1):
		if np.count_nonzero(Model[i]!=0) and np.count_nonzero(membershipList[i]!=0):
			featureSet_i=np.where(Model[i] != 0)[0]
			for j in range(i+1, len(Model)):
				if np.count_nonzero(Model[j]!=0) and np.count_nonzero(membershipList[j]!=0):
					featureSet_j=np.where(Model[j] != 0)[0]
					featureSet+=len(np.intersect1d(featureSet_i , featureSet_j))
	return ((featureSet * 2)/((0.0+NOT_ZERO_DIVISION_SECURITY)+Total_Cluster*(Total_Cluster-1)))

def Feature_Per_Cluster(Model, membershipList,dimension):
	elements=0
	Total_Cluster=No_of_Clusters(Model,membershipList)
	for i in range(len(Model)):
		if np.count_nonzero(Model[i]!=0) and np.count_nonzero(membershipList[i]!=0):
			elements=elements+np.count_nonzero(Model[i])
	FPC=(elements+0.0)/Total_Cluster
	FPC=abs((dimension/2)-FPC)
	return (FPC*2)/(dimension+0.0) #FPC

def Calculate_PSM(Model,membershipList,dimension):
	DNR=dimension_non_redundancy(Model,membershipList)
	FPC=Feature_Per_Cluster(Model, membershipList,dimension)
	return DNR/(dimension+0.0), FPC


def data_clusterDistance(Model,sampleData,SDmax, membershipList):
	compact=0.0
	flag=0
	NoCluster=0

	for m in range(SDmax):
		E_c_dis=0.0
		if np.count_nonzero(Model[m]!=0) and np.count_nonzero(membershipList[m]!=0):
			points=np.count_nonzero(membershipList[m])
			NoCluster=NoCluster+1
			flag=1
			center=Model[m]
			for s in range(len(sampleData)):
				if (membershipList[m][s]!=0):
					sample=sampleData[s]
					DisCal=np.sum(np.absolute(np.subtract(center,sample)))
					E_c_dis=E_c_dis+(membershipList[m][s]*DisCal)
		if (flag==1):
			compact=compact+(E_c_dis/(points+0.0))
			flag=0 

	return compact/(NoCluster+0.0)
