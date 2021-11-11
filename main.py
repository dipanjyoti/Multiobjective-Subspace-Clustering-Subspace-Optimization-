import numpy as np
import random
from pandas import read_table
from sklearn import preprocessing as pp
import time
from util import *
from mutation import *
from objectives import *
from evaluation import *

filename= 'breast.arff' 
filenametrue= 'breast.true' 

sample_size =150
NoOfClasses =2
SDmax=32
NoOfIteration=2001


population_size=10
Iteration=0 
F1_len=0

Model_dict={}
Genotype_dict={}
population=[]
population_new=[]
population_final=[]
solution=[]

trueSubspace=[]
trueSubspaceF=[]
trueCluster=[]
dataTrue =[]
sampleData=[]

#Ivalue=2 *SDmax

start_time = time.clock()
dataframe = read_table(filename, sep=',',header=None)
data = np.array(dataframe)
dataCount=len(data)
data = data[:,0:(len(data[0])-1)]
dimension=len(data[0])
SS = pp.StandardScaler(copy=True, with_mean=True, with_std=True)   
scaled_All = SS.fit_transform(data) 
sampleData=random.sample(scaled_All, sample_size) #Randomly selected sample data of size sample_size 

with open(filenametrue, 'r') as f:
    dataTrue = [[int(num) for num in line.split(' ')] for line in f]

for i in range(len(dataTrue)):
	trueSubspaceF.append(dataTrue[i][:dimension])
	trueCluster.append(dataTrue[i][dimension+1:(len(dataTrue[i]))]) 
trueSubspaceF = np.array(trueSubspaceF)

for i in range (len(trueSubspaceF)):
	index=np.where(trueSubspaceF[i] == 1)[0]
	trueSubspace.append(index)

#Model Initialization
for i in range(population_size):
	Model=np.zeros((SDmax, dimension))
	Genotype=np.zeros((SDmax, dimension))
	Model_dict[i]=Model
	Genotype_dict[i]=Genotype

for i in range(population_size):
	population.append((0,0))
	population_new.append((0,0))
	solution.append((0,0))

for i in range(2*population_size):
	population_final.append((0,0))

Ivalue=max(SDmax,NoOfClasses*(dimension/2))

#Ivalue=(max(SDmax,NoOfClasses*(dimension/2)))/2 #For synthetic data sets , specifically for dimension data set

for i in range(0,len(Model_dict)):
	Model_dict[i], Genotype_dict[i]=Initial_Model(list(Model_dict.values()[i]),list(Genotype_dict.values()[i]),sampleData,SDmax,sample_size,dimension,Ivalue)

def GenerateSolution(Fonts):
    global F1_len
    k=population_size
    distance=[]
    i=0
    x=0
    F=[]
    F=Fonts
    while True:
	    if i>=len(F):
		    return
	    else:
		    if len(F[i])>k:
			    distance=crowding_distance(F[i])
			    e=dict()
			    
			    for j in range(len(F[i])):
				    e[distance[j]]=j

			    distance.sort(reverse=True)
	
			    for j in range (k):
				    solution[x]=F[i][e.get(distance[j])]
				    x=x+1
			    break
		    else:
			    for j in range(len(F[i])):
				    solution[x]=F[i][j]
				    x=x+1
			    k=k-len(F[i])
	    if i==0:
		    F1_len=len(F[0])
	    i=i+1

i=0
while i<population_size:
    Model_child, genotype_child=OneChild1(list(Model_dict.values())[i],list(Genotype_dict.values())[i],sampleData,SDmax,sample_size,dimension,Ivalue)
    Modelnew, Genotypenew=OneChild2(Model_child,genotype_child,sampleData,SDmax,sample_size,dimension,Ivalue)
    membershipList= membershipDegree(Modelnew,sampleData,dimension) 
    FNR, FPC= Calculate_PSM(Modelnew, membershipList,dimension)
    ICD= data_clusterDistance(Modelnew,sampleData,SDmax, membershipList) #Intra Cluster Distance
    population[i]=(FNR+FPC,ICD) #Putting solution for crowding distance
    Model_dict[i]=Modelnew		#Model corresponding to iteration/(XB,PBM) pair
    Genotype_dict[i]=Genotypenew #genotype corresponding to iteration/(XB,PBM) pair
    i=i+1


while Iteration<NoOfIteration:

	if (Iteration%100==0):
		print ("Iteration",Iteration)

	randSample=(random.randint(0,sample_size-1)) # changing samole data one at a time
	randData=(random.randint(0,dataCount-1))
	sampleData[randSample]=scaled_All[randData]

	Model_dict_new={}
	Genotype_dict_new={}
	Model_final_dict={}
	Genotype_final_dict={}


	for i in range(population_size):
		#Prob_new=random.random()
		Model_new, Genotype_new=OneChild1(list(Model_dict.values())[i],list(Genotype_dict.values())[i],sampleData,SDmax,sample_size,dimension,Ivalue)
		Modelnew, Genotypenew=OneChild2(Model_new,Genotype_new,sampleData,SDmax,sample_size,dimension,Ivalue)
		membershipList= membershipDegree(Modelnew,sampleData,dimension)
		FNR, FPC= Calculate_PSM(Modelnew, membershipList,dimension)
		ICD= data_clusterDistance(Modelnew,sampleData,SDmax, membershipList)
		population_new[i%population_size]=(FNR+FPC,ICD)
		Model_dict_new[i%population_size]=Modelnew
		Genotype_dict_new[i%population_size]=Genotypenew

	for k in range(population_size):
		population_final[k]=population[k]
		Model_final_dict[k]=Model_dict[k]
		Genotype_final_dict[k]=Genotype_dict[k]
		population_final[k+population_size]=population_new[k]
		Model_final_dict[k+population_size]=Model_dict_new[k]
		Genotype_final_dict[k+population_size]=Genotype_dict_new[k]

	F=non_dominating(population_final,population_size)
	GenerateSolution(F)
	for k in range(population_size):
		index=population_final.index(solution[k])
		Model_dict[k]=Model_final_dict[index]
		population[k]=population_final[index]
		Genotype_dict[k]=Genotype_final_dict[index]
	Iteration=Iteration+1

Total_Time=(time.clock() - start_time)


F_measure_All=[]
Accuracy_All=[]
CE_All=[]
RNIA_All=[]
Entropy_All=[]
Avgdim_All=[]
Total_Cluster_All=[]

for i in range(len(Model_dict)):
	membershipList= membershipDegree(list(Model_dict.values())[i],scaled_All,dimension)
	F_measure, Accuracy, CE, RNIA, Entropy, Avgdim, Total_Cluster= evaluation(list(Model_dict.values())[i],membershipList,trueCluster,trueSubspace,scaled_All,dimension)
	F_measure_All.append(F_measure)
	Accuracy_All.append(Accuracy)
	CE_All.append(CE)
	RNIA_All.append(RNIA)
	Entropy_All.append(Entropy)
	Avgdim_All.append(Avgdim)
	Total_Cluster_All.append(Total_Cluster)

print 
print 'sample_size=',sample_size
print 'SDmax=', SDmax
print 'Iteration=', NoOfIteration
print 
print 'F_measure: MAX, MIN, ALL', max(F_measure_All), min(F_measure_All), F_measure_All
print 
print 'Accuracy: MAX, MIN, ALL', max(Accuracy_All), min(Accuracy_All), Accuracy_All
print 
print 'CE: MAX, MIN, ALL', max(CE_All), min(CE_All), CE_All
print 
print 'RNIA: MAX, MIN, ALL', max(RNIA_All), min(RNIA_All), RNIA_All
print 
print 'Entropy: MAX, MIN, ALL', max(Entropy_All), min(Entropy_All), Entropy_All
print 
print 'Coverage=', 1
print 
print 'NumClusters: MAX, MIN, ALL', max(Total_Cluster_All), min(Total_Cluster_All), Total_Cluster_All
print 
print 'AvgDim: MAX, MIN, ALL', max(Avgdim_All), min(Avgdim_All), Avgdim_All
print 
print 'run_Time=',Total_Time
print 

print round(max(F_measure_All), 2), round(min(F_measure_All), 2), round(max(Accuracy_All), 2), round(min(Accuracy_All), 2), round(max(CE_All), 2), round(min(CE_All), 2), round(max(RNIA_All), 2), round(min(RNIA_All), 2), round(max(Entropy_All), 2), round(min(Entropy_All), 2), 1.00, 1.00, round(max(Total_Cluster_All), 2), round(min(Total_Cluster_All), 2), round(max(Avgdim_All), 2), round(min(Avgdim_All), 2), Total_Time, Total_Time
