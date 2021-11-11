import numpy as np
import random
import math

def OneChild1(M ,G,sampleData, SDmax, sample_size,dimension,Ivalue):

	weight=0
	prob=random.random()

	Model_child=[]
	genotype_child=[]
	clusterRow=[]
	nonClusterRow=[]
	sample=[]

	for i in range(SDmax):
		Model_child.append([])
		genotype_child.append([])
		for j in range(dimension):
			Model_child[i].append(M[i][j])
			genotype_child[i].append(G[i][j])

	Model_child = np.array(Model_child)
	genotype_child = np.array(genotype_child)
	weight=genotype_child.sum()

	for m in range(SDmax):
		if np.count_nonzero(genotype_child[m]!=0):
			clusterRow.append(m)
			np.random.shuffle(clusterRow)
		else:
			nonClusterRow.append(m)
			np.random.shuffle(nonClusterRow)

	if weight>=2*Ivalue:
		r1=(random.randint(0,SDmax-1))
		r2=(random.randint(0,dimension-1))
		while genotype_child[r1][r2] == 0:
			r1=(random.randint(0,SDmax-1))
			r2=(random.randint(0,dimension-1))
		genotype_child[r1][r2]=genotype_child[r1][r2]-1
		if genotype_child[r1][r2]==0:
			Model_child[r1][r2]=0.0

	rSample=(random.randint(0,sample_size-1))
	features = random.sample(range(0, dimension-1), random.choice(range(1, int(math.ceil(dimension/2.0))))) 

	for i in range(len(features)):
		sample.append(sampleData[rSample][features[i]])

	if (len(features)/(weight+(len(features))))>prob:
		for i in range(len(features)):
			Model_child[nonClusterRow[0]][features[i]]=sample[i]
			genotype_child[nonClusterRow[0]][features[i]]=genotype_child[nonClusterRow[0]][features[i]]+1
	else:
		a=0
		newCluster=clusterRow[0]
		for i in range (len(clusterRow)):
			common=len(np.intersect1d((np.nonzero(Model_child[clusterRow[i]])),features))
			if common>a:
				a=common
				newCluster=clusterRow[i]

		for i in range(len(features)):
			if Model_child[newCluster][features[i]]==0:
				Model_child[newCluster][features[i]]=sample[i]
				genotype_child[newCluster][features[i]]=genotype_child[newCluster][features[i]]+1
			else:
				Model_child[newCluster][features[i]]=(Model_child[newCluster][features[i]]+sample[i])/2
				genotype_child[newCluster][features[i]]=genotype_child[newCluster][features[i]]+1

	return Model_child, genotype_child

def bionomial(L):
	mu=0.005
	s= np.random.binomial(L, mu)
	return max(s,1)

def replace_by(count,flag):
	center=np.zeros(count)
	return center

def cluster_center_change(center_num):
	tempval=np.random.normal(center_num,0.2,1)
	return tempval[0]

def randomSum(value):
	a=value
	num=[]
	if (value<=2):
		num.append(value)

	while (a>2):
		val1=random.choice(range(1,a))
		num.append(val1)
		a=value-sum(num)
		if a<=2:
			num.append(a)
			break
	return num

def OneChild2(M ,G,sampleData, SDmax, sample_size,dimension,Ivalue):

	weight=0

	Model_child=[]
	genotype_child=[]
	clusterRow=[]
	nonClusterRow=[]

	for i in range(SDmax):
		Model_child.append([])
		genotype_child.append([])
		for j in range(dimension):
			Model_child[i].append(M[i][j])
			genotype_child[i].append(G[i][j])

	Model_child = np.array(Model_child)
	genotype_child = np.array(genotype_child)
	weight=genotype_child.sum()

	for m in range(SDmax):
		if np.count_nonzero(genotype_child[m]!=0):
			clusterRow.append(m)
			np.random.shuffle(clusterRow)
		else:
			nonClusterRow.append(m)
			np.random.shuffle(nonClusterRow)

	if weight>=2*Ivalue:
		r1=(random.randint(0,SDmax-1))
		r2=(random.randint(0,dimension-1))
		while genotype_child[r1][r2] == 0:
			r1=(random.randint(0,SDmax-1))
			r2=(random.randint(0,dimension-1))
		genotype_child[r1][r2]=genotype_child[r1][r2]-1
		if genotype_child[r1][r2]==0:
			Model_child[r1][r2]=0.0

	value1=bionomial(len(clusterRow))

	for i in range(value1): 

		center_num=clusterRow[i]
		gene_count=genotype_child[center_num].sum()
		value2=bionomial(gene_count)

		for j in range(value2):
			if np.count_nonzero(Model_child[center_num])==0:
				break
			if (np.count_nonzero(Model_child[center_num])==1):
				value3=1 * (random.choice([-1,1]))
			else:
				value3=(random.choice(range(1, np.count_nonzero(Model_child[center_num])))) * (random.choice([-1,1]))

			NZ_features = np.nonzero(Model_child[center_num])

			if len(NZ_features[0])==1:
				rand_features=NZ_features[0]
			else:
				rand_features = random.sample(NZ_features[0], abs(value3))

			if (value3<0):
				num=randomSum(abs(value3))
				b=0
				for p in range (len(num)):
					for q in range(num[p]):
						rclust=random.randint(1,len(clusterRow))
						Model_child[rclust][rand_features[q+b]]==0
						genotype_child[rclust][rand_features[q+b]]==0
					b=b+num[p]
			else:
				num=randomSum(abs(value3))
				b=0
				for p in range (len(num)):
					center_no=random.randint(1,len(clusterRow))
					for k in range(num[p]):
						Model_child[center_num][rand_features[k+b]]=cluster_center_change(Model_child[center_num][rand_features[k+b]])
						genotype_child[center_num][rand_features[k+b]]=genotype_child[center_num][rand_features[k+b]]-1

						if Model_child[center_no][rand_features[k+b]]==0:
							Model_child[center_no][rand_features[k+b]]=Model_child[center_num][rand_features[k+b]]
							genotype_child[center_no][rand_features[k+b]]=genotype_child[center_no][rand_features[k+b]]+1
						else:
							Model_child[center_no][rand_features[k+b]]=(Model_child[center_no][rand_features[k+b]] + Model_child[center_num][rand_features[k+b]])/2
							genotype_child[center_no][rand_features[k+b]]=genotype_child[center_no][rand_features[k+b]]+1

						if genotype_child[center_num][rand_features[k+b]]==0:
							Model_child[center_num][rand_features[k+b]]=0
					b=b+num[p]

	return Model_child, genotype_child







# def OneChild2_1(M ,G,sampleData, SDmax, sample_size,dimension,Ivalue): #original

# 	weight=0

# 	Model_child=[]
# 	genotype_child=[]
# 	clusterRow=[]
# 	nonClusterRow=[]

# 	for i in range(SDmax):
# 		Model_child.append([])
# 		genotype_child.append([])
# 		for j in range(dimension):
# 			Model_child[i].append(M[i][j])
# 			genotype_child[i].append(G[i][j])

# 	Model_child = np.array(Model_child)
# 	genotype_child = np.array(genotype_child)
# 	weight=genotype_child.sum()

# 	for m in range(SDmax):
# 		if np.count_nonzero(genotype_child[m]!=0):
# 			clusterRow.append(m)
# 			np.random.shuffle(clusterRow)
# 		else:
# 			nonClusterRow.append(m)
# 			np.random.shuffle(nonClusterRow)

# 	if weight>=2*Ivalue:
# 		r1=(random.randint(0,SDmax-1))
# 		r2=(random.randint(0,dimension-1))
# 		while genotype_child[r1][r2] == 0:
# 			r1=(random.randint(0,SDmax-1))
# 			r2=(random.randint(0,dimension-1))
# 		genotype_child[r1][r2]=genotype_child[r1][r2]-1
# 		if genotype_child[r1][r2]==0:
# 			Model_child[r1][r2]=0.0

# 	value1=bionomial(len(clusterRow))

# 	for i in range(value1): 

# 		center_num=clusterRow[i]
# 		gene_count=genotype_child[center_num].sum()
# 		value2=bionomial(gene_count)

# 		for j in range(value2):
# 			if np.count_nonzero(Model_child[center_num])==0:
# 				break
# 			if (np.count_nonzero(Model_child[center_num])==1):
# 				value3=1 * (random.choice([-1,1]))
# 			else:
# 				value3=(random.choice(range(1, np.count_nonzero(Model_child[center_num])))) * (random.choice([-1,1]))

# 			NZ_features = np.nonzero(Model_child[center_num])

# 			if len(NZ_features[0])==1:
# 				rand_features=NZ_features[0]
# 			else:
# 				rand_features = random.sample(NZ_features[0], abs(value3))

# 			if (value3<0):
# 				num=randomSum(abs(value3))
# 				b=0
# 				for p in range (len(num)):
# 					for q in range(num[p]):
# 						rclust=random.randint(1,len(clusterRow))
# 						Model_child[rclust][rand_features[q+b]]==0
# 						genotype_child[rclust][rand_features[q+b]]==0
# 					b=b+num[p]

# 			else:
# 				center_no=random.randint(1,len(clusterRow))#clusterRow[len(clusterRow)-i-1]
# 				for k in range(len(rand_features)):
# 					Model_child[center_num][rand_features[k]]=cluster_center_change(Model_child[center_num][rand_features[k]])
# 					genotype_child[center_num][rand_features[k]]=genotype_child[center_num][rand_features[k]]-1

# 					if Model_child[center_no][rand_features[k]]==0:
# 						Model_child[center_no][rand_features[k]]=Model_child[center_num][rand_features[k]]
# 						genotype_child[center_no][rand_features[k]]=genotype_child[center_no][rand_features[k]]+1
# 					else:
# 						Model_child[center_no][rand_features[k]]=(Model_child[center_no][rand_features[k]] + Model_child[center_num][rand_features[k]])/2
# 						genotype_child[center_no][rand_features[k]]=genotype_child[center_no][rand_features[k]]+1

# 					if genotype_child[center_num][rand_features[k]]==0:
# 						Model_child[center_num][rand_features[k]]=0

# 	return Model_child, genotype_child







# def OneChild2_0(M ,G,sampleData, SDmax, sample_size,dimension,Ivalue):

# 	weight=0

# 	Model_child=[]
# 	genotype_child=[]
# 	clusterRow=[]
# 	nonClusterRow=[]

# 	for i in range(SDmax):
# 		Model_child.append([])
# 		genotype_child.append([])
# 		for j in range(dimension):
# 			Model_child[i].append(M[i][j])
# 			genotype_child[i].append(G[i][j])

# 	Model_child = np.array(Model_child)
# 	genotype_child = np.array(genotype_child)
# 	weight=genotype_child.sum()

# 	print 'weight2', weight

# 	for m in range(SDmax):
# 		if np.count_nonzero(genotype_child[m]!=0):
# 			clusterRow.append(m)
# 			np.random.shuffle(clusterRow)
# 		else:
# 			nonClusterRow.append(m)
# 			np.random.shuffle(nonClusterRow)

# 	while weight>=2 * Ivalue:
# 		#print 'weight1', weight, 2*Ivalue
# 		r1=(random.randint(0,SDmax-1))
# 		r2=(random.randint(0,dimension-1))
# 		while genotype_child[r1][r2] == 0:
# 			r1=(random.randint(0,SDmax-1))
# 			r2=(random.randint(0,dimension-1))
# 		genotype_child[r1][r2]=genotype_child[r1][r2]-1
# 		if genotype_child[r1][r2]==0:
# 			Model_child[r1][r2]=0.0
# 		weight=genotype_child.sum()


# 	value1=bionomial(len(clusterRow))

# 	#print 'value1',value1

# 	for i in range(value1): #value1

# 		center_num=clusterRow[i]
# 		gene_count=genotype_child[center_num].sum()
# 		value2=bionomial(gene_count)
# 		#print 'value2',value2

# 		for j in range(value2): #value2
# 			if np.count_nonzero(Model_child[center_num])==0:
# 				break
# 			if (np.count_nonzero(Model_child[center_num])==1):
# 				value3=1 * (random.choice([-1,1]))
# 			else:
# 				value3=(random.choice(range(1, np.count_nonzero(Model_child[center_num])))) * (random.choice([-1,1]))

# 			NZ_features = np.nonzero(Model_child[center_num])

# 			if len(NZ_features[0])==1:
# 				rand_features=NZ_features[0]
# 			else:
# 				rand_features = random.sample(NZ_features[0], abs(value3))


# 			if (value3<0):

# 				# np.put(Model_child[center_num], rand_features, replace_by(len(rand_features),0))
# 				# np.put(genotype_child[center_num], rand_features, replace_by(len(rand_features),0))

# 				prob1=random.random()

# 				if prob1>0.5:
# 					for i in range(len(rand_features)):
# 						rclust=random.randint(1,len(clusterRow))
# 						Model_child[rclust][rand_features[i]]==0
# 						genotype_child[rclust][rand_features[i]]==0
# 				else:
# 					np.put(Model_child[center_num], rand_features, replace_by(len(rand_features),0))
# 					np.put(genotype_child[center_num], rand_features, replace_by(len(rand_features),0))

# 			else:
# 				center_no=random.randint(1,len(clusterRow))#clusterRow[len(clusterRow)-i-1]
# 				for k in range(len(rand_features)):
# 					Model_child[center_num][rand_features[k]]=cluster_center_change(Model_child[center_num][rand_features[k]])
# 					#genotype_child[center_num][rand_features[k]]=genotype_child[center_num][rand_features[k]]-1

# 					if Model_child[center_no][rand_features[k]]==0:
# 						Model_child[center_no][rand_features[k]]=Model_child[center_num][rand_features[k]]
# 						genotype_child[center_no][rand_features[k]]=genotype_child[center_no][rand_features[k]]+1
# 					else:
# 						Model_child[center_no][rand_features[k]]=(Model_child[center_no][rand_features[k]] + Model_child[center_num][rand_features[k]])/2
# 						genotype_child[center_no][rand_features[k]]=genotype_child[center_no][rand_features[k]]+1

# 					#if genotype_child[center_num][rand_features[k]]==0:
# 						#Model_child[center_num][rand_features[k]]=0

# 	return Model_child, genotype_child


				# np.put(Model_child[center_num], rand_features, replace_by(len(rand_features),0))
				# np.put(genotype_child[center_num], rand_features, replace_by(len(rand_features),0))

				# prob1=random.random()

				# if prob1>0.5:
				# 	for i in range(len(rand_features)):
				# 		rclust=random.randint(1,len(clusterRow))
				# 		Model_child[rclust][rand_features[i]]==0
				# 		genotype_child[rclust][rand_features[i]]==0
				# else:
				# 	np.put(Model_child[center_num], rand_features, replace_by(len(rand_features),0))
				# 	np.put(genotype_child[center_num], rand_features, replace_by(len(rand_features),0))	