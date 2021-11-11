import numpy as np
import pandas as pd
import random
import itertools
from objectives import No_of_Clusters
NOT_ZERO_DIVISION_SECURITY = 1e-10

def _validity_cluster_checking(found_clusters_effective,
                               threshold_cluster_validity=0.0):
    return found_clusters_effective >= threshold_cluster_validity

def _mapped(contingency_table):
    mapped_clusters = (contingency_table.T * 1. / contingency_table.sum(1)).T
    return mapped_clusters == mapped_clusters.max(0)

def compute_only_f1(contingency_table,
                    valid_clusters,
                    mapped_clusters):
    num = mapped_clusters * contingency_table
    num = num.loc[:, valid_clusters]
    num = num.sum(1)
    denum_recall = contingency_table.sum(1)
    rec = num * 1. / (denum_recall + NOT_ZERO_DIVISION_SECURITY)
    denum_precision = mapped_clusters * contingency_table.sum(0) * valid_clusters
    denum_precision = denum_precision.sum(1)
    precis = num * 1. / (denum_precision + NOT_ZERO_DIVISION_SECURITY)
    denum = rec + precis
    num = 2 * rec * precis
    return sum(num * 1.0 / (denum + NOT_ZERO_DIVISION_SECURITY)) * 1. / (len(num) + NOT_ZERO_DIVISION_SECURITY)

def f1(contingency_table,
       threshold_cluster_validity=0.0):
    mapped_clusters = _mapped(contingency_table)
    found_clusters_effective = contingency_table.sum(0)
    valid_clusters = _validity_cluster_checking(found_clusters_effective, threshold_cluster_validity)
    return compute_only_f1(contingency_table, valid_clusters, mapped_clusters)

def compute_only_entropy(contingency_table,
                         valid_clusters):
    contingency_table = contingency_table.loc[:, valid_clusters]
    found_clusters_effective = contingency_table.sum(0)
    p_h_in_c = contingency_table * 1. / (found_clusters_effective + NOT_ZERO_DIVISION_SECURITY)
    log_p_h_in_c = np.log(p_h_in_c)
    pre_ec = -1. * p_h_in_c * log_p_h_in_c
    pre_ec = pre_ec.fillna(0)
    ec = pre_ec.sum(0)
    num = (ec * found_clusters_effective).sum()
    denum = found_clusters_effective.sum() * np.log(len(contingency_table.index))
    return 1. - num * 1. / denum

def entropy(contingency_table,
            threshold_cluster_validity=0.0):
    #contingency_table = pd.crosstab(cluster_hidden, cluster_found)
    valid_clusters = _validity_cluster_checking(contingency_table.sum(0), threshold_cluster_validity)
    return compute_only_entropy(contingency_table, valid_clusters)

def compute_only_accuracy(contingency_table, valid_clusters, found_clusters_effective):
    best_matching_hidden_cluster = contingency_table == contingency_table.max(0)
    best_matching_hidden_cluster_weight = 1. / best_matching_hidden_cluster.sum(0)
    correctly_predicted_objects = contingency_table * best_matching_hidden_cluster * best_matching_hidden_cluster_weight
    correctly_predicted_objects *= valid_clusters
    return sum(correctly_predicted_objects.sum(0)) * 1. / (sum(found_clusters_effective)+NOT_ZERO_DIVISION_SECURITY)

def accuracy(contingency_table, threshold_cluster_validity=0.0):
    #contingency_table = pd.crosstab(cluster_hidden, cluster_found)
    found_clusters_effective = contingency_table.sum(0)
    valid_clusters = _validity_cluster_checking(found_clusters_effective, threshold_cluster_validity)
    return compute_only_accuracy(contingency_table, valid_clusters, found_clusters_effective)

# Start of Functions Computes a max weight perfect matching in a bipartite graph
def improveLabels(val):
    """ change the labels, and maintain minSlack.
    """
    for u in S:
        lu[u] -= val
    for v in V:
        if v in T:
            lv[v] += val
        else:
            minSlack[v][0] -= val

def improveMatching(v):
    """ apply the alternating path from v to the root in the tree.
    """
    u = T[v]
    if u in Mu:
        improveMatching(Mu[u])
    Mu[u] = v
    Mv[v] = u

def slack(u,v): return lu[u]+lv[v]-w[u][v]

def augment():
    """ augment the matching, possibly improving the lablels on the way.
    """
    while True:
        # select edge (u,v) with u in S, v not in T and min slack
        ((val, u), v) = min([(minSlack[v], v) for v in V if v not in T])
        assert u in S
        if val>0:
            improveLabels(val)
        # now we are sure that (u,v) is saturated
        assert slack(u,v)==0
        T[v] = u                            # add (u,v) to the tree
        if v in Mv:
            u1 = Mv[v]                      # matched edge,
            assert not u1 in S
            S[u1] = True                    # ... add endpoint to tree
            for v in V:                     # maintain minSlack
                if not v in T and minSlack[v][0] > slack(u1,v):
                    minSlack[v] = [slack(u1,v), u1]
        else:
            improveMatching(v)              # v is a free vertex
            return

def maxWeightMatching(weights):
    """ given w, the weight matrix of a complete bipartite graph,
        returns the mappings Mu : U->V ,Mv : V->U encoding the matching
        as well as the value of it.
    """
    global U,V,S,T,Mu,Mv,lu,lv, minSlack, w
    w  = weights
    n  = len(w)
    U  = V = range(n)
    lu = [ max([w[u][v] for v in V]) for u in U]  # start with trivial labels
    lv = [ 0                         for v in V]
    Mu = {}                                       # start with empty matching
    Mv = {}
    while len(Mu)<n:
        free = [u for u in V if u not in Mu]      # choose free vertex u0
        u0 = free[0]
        S = {u0: True}                            # grow tree from u0 on
        T = {}
        minSlack = [[slack(u0,v), u0] for v in V]
        augment()
    #                                    val. of matching is total edge weight
    val = sum(lu)+sum(lv)
    return (Mu, Mv, val)

def compute_CE_RNIA(Total_Cluster, Model_Selected, PredCluster, trueCluster,mydata_list_of_list,membershipList,dimension,trueSubspace):

	dimensionSet=[]
	PredLevelModel=[]
	dimensionSetIndex=[]

	for i in range (Total_Cluster):
		dimensionSetIndex.append([])

	for s in range (len(mydata_list_of_list)):
		for m in range (len(Model_Selected)):
			if membershipList[m][s]==1:
				PredLevelModel.append(m)
				break

	for i in range (Total_Cluster):
		X=PredCluster[i][0]
		M=PredLevelModel[X]
		dimensionSet.append(Model_Selected[M])

	for i in range (Total_Cluster):
		for d in range (dimension):
			if dimensionSet[i][d]!=0:
				dimensionSetIndex[i].append(d)

	size=max(len(trueCluster),len(PredCluster))

	confusionMatrix=[]

	for i in range(size):
		confusionMatrix.append([])
		for j in range(size):
			confusionMatrix[i].append(0)

	for i in range (len(PredCluster)):
		for j in range (len(trueCluster)):
			same=len(set(PredCluster[i]) & set(trueCluster[j]))
			common= len(set(dimensionSetIndex[i]) & set(trueSubspace[j]))
			confusionMatrix[i][j]=common*same

	confusionMatrix = [[confusionMatrix[j][i] for j in range(len(confusionMatrix))] for i in range(len(confusionMatrix[0]))]
	D_max= (list(maxWeightMatching(confusionMatrix))[-1])
	return D_max, confusionMatrix, dimensionSetIndex

# def calculate_Union(PredCluster, PredSubspace): #Only for non-overlapping but not for overlapping

# 	Pred_All_U=[]
# 	True_All_U=[]
# 	for i in range(len(PredCluster)):
# 		for k in itertools.product(PredCluster[i],PredSubspace[i]):
# 			Pred_All_U.append(k)
# 	Pred_All_U=set(Pred_All_U)

# 	for i in range(len(trueCluster)):
# 		for k in itertools.product(trueCluster[i],trueSubspace[i]):
# 			True_All_U.append(k)
# 	True_All_U=set(True_All_U)

#    	All_U= Pred_All_U.union(True_All_U)  #set(j).union(set(k))
#    	U=len(All_U)
#    	return U


def calculate_Union(PredCluster, PredSubspace,trueSubspace,trueCluster): # More Accurate for both overlapping and non overlapping

	Pred_All_U=[]
	True_All_U=[]

	for i in range(len(PredCluster)):
		for k in itertools.product(PredCluster[i],PredSubspace[i]):
			Pred_All_U.append(k)

	for i in range(len(trueCluster)):
		for k in itertools.product(trueCluster[i],trueSubspace[i]):
			True_All_U.append(k)

	for i in range (len(True_All_U)):
		if True_All_U[i] not in Pred_All_U:
			Pred_All_U.append(True_All_U[i])

   	U=len(Pred_All_U)
   	return U



def evaluation(Model_Selected,membershipList,trueCluster,trueSubspace,mydata_list_of_list,dimension):
	membershipListNonZero=[]
	PredLevel=[]
	PredCluster=[]

	Model_Selected=np.asarray(Model_Selected)
	membershipList=np.asarray(membershipList)
	Total_Cluster= No_of_Clusters(Model_Selected,membershipList)
	Model_Selected=Model_Selected.tolist()

	for i in range (len(membershipList)):
		if np.count_nonzero(membershipList[i]!=0):
			membershipListNonZero.append(membershipList[i])

	for s in range (len(mydata_list_of_list)):
		for m in range (len(membershipListNonZero)): 
			if membershipListNonZero[m][s]==1:
				PredLevel.append(m)
				break

	for i in range(Total_Cluster):
		PredCluster.append([])

	for i in range(len(mydata_list_of_list)):
		PredCluster[PredLevel[i]].append(i)

	contingency_table=[]

	for i in range(len(trueCluster)):
		contingency_table.append([])

	for i in range (len(trueCluster)):
		truesame=[]
		for j in range (len(PredCluster)):
			truesame.append(len(set(trueCluster[i]) & set(PredCluster[j])))
			contingency_table[i]=truesame
	contingency_table1 = pd.DataFrame(contingency_table)
	contingency_table=contingency_table1
	D_max, confusionMatrix, PredSubspace=compute_CE_RNIA(Total_Cluster, Model_Selected, PredCluster, trueCluster,mydata_list_of_list,membershipList,dimension,trueSubspace)

	F_measure=f1(contingency_table, threshold_cluster_validity=0.0)
	Entropy = entropy(contingency_table, threshold_cluster_validity=0.0)
	Accuracy=accuracy(contingency_table,threshold_cluster_validity=0.0)
	I=0
	for i in range(Total_Cluster):
		I=I+sum(confusionMatrix[i])
	U=calculate_Union(PredCluster, PredSubspace,trueSubspace,trueCluster)
	#U1=len(mydata_list_of_list)*dimension # U value will be different for syntetic dataset
	CE=(float)(D_max)/(U+0.0)
	RNIA=(float)(I)/(U+0.0)

	Avgdim=0.0
	for i in range (Total_Cluster):
		Avgdim=Avgdim+len(PredSubspace[i])
	Avgdim=Avgdim/(Total_Cluster+0.0)
	return F_measure, Accuracy, CE, RNIA, Entropy, Avgdim, Total_Cluster

