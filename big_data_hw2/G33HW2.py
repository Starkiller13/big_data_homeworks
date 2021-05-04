from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from operator import add
import math
import time

def strToTuple(line):
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1]))    

def map1(elem, smp, cl_s, t, k):
    x = elem[0]
    y = elem[1]
    c_sum = sum([dist(x,p[0]) for p in [x for x in smp if x[1]==y]])
    l_sums = [sum([dist(x,p[0]) for p in [x for x in smp if x[1]==j]])/min(t,cl_s[j]) for j in range(0,k) if j!=y]
    a = c_sum/min(t,cl_s[y])
    b = min(l_sums)
    return (0,(b-a)/max(b,a))

def dist(x,y):
    l = [(x[i]-y[i])**2 for i in range(len(x))]
    l = math.sqrt(sum(l))
    return l

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
    
    fullClustering = sc.textFile(data_path, minPartitions=10).cache()
    fullClustering = fullClustering.map(strToTuple)
    
    C = sorted(fullClustering.map(lambda x: x[1]).countByValue().items()) 
    C = [C[i][1] for i in range(len(C))]
    
    sharedClusterSize = sc.broadcast(C)
    
    samples = fullClustering.map(lambda x : x if rand.random()<=min(t/sharedClusterSize.value[x[1]],1) else None).filter(lambda x: x!=None).collect()
    
    clusteringSample = sc.broadcast(samples)
    
    start1 = time.time_ns()
    
    s = []
    for i in range(t):
        x = samples[i][0]
        y = samples[i][1]
        c_sum = sum([dist(x,p[0]) for p in [x for x in samples if x[1]==y]])
        l_sums = [sum([dist(x,p[0]) for p in [x for x in samples if x[1]==j]])/C[j] for j in range(0,k) if j!=y]
        a = c_sum/min(t,C[y])
        b = min(l_sums)
        s.append((b-a)/max(b,a))
    #samples = samples.map(lambda x: map1(x,clusteringSample.value, sharedClusterSize.value, t, k)).collect()
    exactSilhSample = sum(s)/t
    
    end1 = time.time_ns()
    
    start0 = time.time_ns()
    
    N = fullClustering.count()
    fullClustering = (fullClustering.map(lambda x: map1(x,clusteringSample.value, sharedClusterSize.value, t, k))
            .reduceByKey(add))
    approxSilhFull = fullClustering.take(1)[1]/N
    
    end0 = time.time_ns()
    
    print("Value of approxSilhFull = %f"%(approxSilhFull))
    print("Time to compute approxSilhFull = %d ms"%(int((end0-start0)/1000000)))
    print("Value of exactSilhSample = %f"%(exactSilhSample))
    print("Time to compute exactSilhSample = %d ms"%(int((end1-start1)/1000000)))
    
    
if __name__ == "__main__":
    main()