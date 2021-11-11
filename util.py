import numpy as np
import pygmo as pg
import random

def Initial_Model(Model, Genotype, sampleData, SDmax, sample_size, dimension,Ivalue):

	Model_Init=[]
	Genotype_Init=[]

	for i in range(SDmax):
		Model_Init.append([])
		Genotype_Init.append([])
		for j in range(dimension):
			Model_Init[i].append(Model[i][j])
			Genotype_Init[i].append(Genotype[i][j])

	random_list=random.sample(range(0,SDmax),SDmax)
	count=0
	flag=0
	while count<2*Ivalue: #2*Ivalue
		for i in random_list:
			for j in range(0,dimension):
				prob=random.random()
				if prob>0.5:
					x1=(random.randint(0,sample_size-1))
					x2=(random.randint(0,dimension-1))
					Model_Init[i][j]=sampleData[x1][x2]
					Genotype_Init[i][j]=Genotype_Init[i][j]+1
					count=count+1
					if count>=2*Ivalue: #2*Ivalue
						flag=1
						break
			if flag==1:
				break
		if flag==1:
			break
	return (Model_Init,Genotype_Init)


def membershipDegree(c,s,dimension):
	C = len(c)
	S = len(s)
	mem_array = np.zeros((C,S),dtype=int)
	#c = np.array(c)

	for i in range(S):
		sample = np.array(s[i])
		center = np.zeros(dimension)
		minm = np.sum(np.absolute(np.subtract(center,sample)))
		ind = 0
		for k in range(C):
			center = np.array(c[k])
			min_temp = np.sum(np.absolute(np.subtract(center,sample)))
			if(min_temp<minm):
				minm=min_temp
				ind=k
		mem_array[ind][i]=1
                          
	return mem_array

def non_dominating(P,population_size):
	population1=[]
	population1=P

	F_one_set=[]
	for i in range(2*population_size):
		F_one_set.append([])
	ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=population1) #ndf, dl, dc, 
	for i in range(2*population_size):
		F_one_set[ndr[i]].append(population1[i])
	F_one_set2 = [x for x in F_one_set if x != []]
	return F_one_set2

def Sort(F,i):
	return (sorted(F,key=lambda x:x[i]))
def Max(F,i):
	return (max(F,key=lambda x:x[i])[i])
def Min(F,i):
	return (min(F,key=lambda x:x[i])[i])

def crowding_distance(Font):
	F=[]
	F=Font
	distance=[]
	for i in range(len(F)):
		distance.append(0)
	noOfObjectives=2

	for i in range (noOfObjectives):
		F_new=Sort(F,i)
		if Max(F_new,i)==Min(F_new,i):
			continue
		distance[0]=99999.0
		distance[len(F)-1]=99999.0
		for j in range (1,len(F)-1):
			distance[j]+=(float)(F_new[j+1][i]-F_new[j-1][i])/((Max(F_new,i)-Min(F_new,i))+0.0)
	return distance
