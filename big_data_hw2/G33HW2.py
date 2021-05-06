from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from operator import add
from math import sqrt
import time


def strToTuple(line):
    """Input parsing function"""
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1]))    

def map1(elem, smp, cl_s, t, k, S_):
    x = elem[0]
    y = elem[1]
    c_sum = sum([dist(x,p) for p in S_[y]])
    l_sums = [sum([dist(x,p) for p in S_[j]])/min(t,cl_s[j]) for j in range(0,k) if j!=y]
    a = c_sum/min(t,cl_s[y]-1)
    b = min(l_sums)
    return (0,(b-a)/max(b,a))

def dist(x,y):
    l = sum([(x[i]-y[i])**2 for i in range(len(x))])
    return l

def Bernoulli(p, n=1): 
    return rand.choices([1,0], weights=[p, (1-p)], k=n) 

def sampling(data,t,sharedClusterSize): 
    t_i = [] 
    for C_i in sharedClusterSize: 
        t_i.append(min(t, C_i)) 
    samples = (data.map(lambda pair : pair if Bernoulli(t_i[pair[1]]/sharedClusterSize[pair[1]]) else None) 
                   .filter(lambda pair: pair != None)) 
    return samples 

def main():
    rand.seed(42)
    assert len(sys.argv) == 4, "Usage: python G33HW1.py <file_name> <k> <t>"
    
    conf = SparkConf().setAppName('G33HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    
    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)
    
    t = sys.argv[3]
    assert t.isdigit(), "t must be an integer"
    t = int(t)
    
    fullClustering = sc.textFile(data_path, minPartitions=8).cache()
    fullClustering = fullClustering.map(strToTuple)
    
    C = sorted(fullClustering.map(lambda x: x[1]).countByValue().items())
     
    C = [C[i][1] for i in range(len(C))]
    
    sharedClusterSize = sc.broadcast(C)
    
    samples = fullClustering.map(lambda x : x if rand.random()<=min(t/sharedClusterSize.value[x[1]],1) else None).filter(lambda x: x!=None)
    S_ = sorted(samples.map(lambda x: (x[1],x[0])).groupByKey().collect())
    S_ = [list(x[1]) for x in S_]
    
    C_s = sorted(samples.map(lambda x: x[1]).countByValue().items()) 
    C_s = [C_s[i][1] for i in range(len(C_s))]
    samples = samples.collect()
    
    clusteringSample = sc.broadcast(samples)
    
    start1 = time.time_ns()
    
    s = []
    for i in range(len(samples)):
        x = samples[i][0]
        y = samples[i][1]
        c_sum = sum([dist(x,p) for p in S_[y]])
        l_sums = [sum([dist(x,p) for p in S_[j]])/C_s[j] for j in range(0,k) if j!=y]
        a = c_sum/(C_s[y]-1)
        b = min(l_sums)
        s.append((b-a)/max(b,a))
    exactSilhSample = sum(s)/len(s)
    
    end1 = time.time_ns()
    
    start0 = time.time_ns()
    
    N = fullClustering.count()
    fullClustering = (fullClustering.map(lambda x: map1(x,clusteringSample.value, sharedClusterSize.value, t, k,S_))
            .reduceByKey(add))
    approxSilhFull = float(fullClustering.collect()[0][1])/N
    
    end0 = time.time_ns()
    
    print("Value of approxSilhFull = %f"%(approxSilhFull))
    print("Time to compute approxSilhFull = %d ms"%(int((end0-start0)/1000000)))
    print("Value of exactSilhSample = %f"%(exactSilhSample))
    print("Time to compute exactSilhSample = %d ms"%(int((end1-start1)/1000000)))
    
    
    
if __name__ == "__main__":
    main()