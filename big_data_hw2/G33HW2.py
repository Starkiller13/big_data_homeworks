from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from operator import add
from math import sqrt
import time
import numpy as np


def strToTuple(line):
    """Input parsing function"""
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1]))    

def map1(elem, smp, cl_s, t):
    x = elem[0]
    y = elem[1]
    dists = [(p[1],dist(x,p[0])) for p in smp.value]
    sums = [0 for i in range(len(cl_s.value))]
    for i in dists:
        sums[i[0]]+=i[1]
    sums = [j/min(t,cl_s.value[i]) for i,j in enumerate(sums)]
    print(sums)
    a = sums.pop(y)
    b = min(sums)
    return (0,(b-a)/max(abs(b),abs(a)))

def bigBrainMap(elem, c_sums, c_prods, s_sizes, c_sizes, t):
    x = elem[0]
    y = elem[1]
    x_sq = dot(x,x)
    w = [(-2*dot(x,c_sums[i]) + c_prods[i] + s_sizes[i]*x_sq)/s_sizes[i] for i in range(len(c_sizes.value))]
    a = w.pop(y)
    b = min(w)
    return (0,(b-a)/max(a,b))

def dot(x,y):
    return sum([x[i]*y[i] for i in range(len(x))])

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
    N = fullClustering.count()

    "Pointest P cluster sizes"
    C = sorted(fullClustering.map(lambda x: x[1]).countByValue().items())
    C = [C[i][1] for i in range(len(C))]
    
    sharedClusterSize = sc.broadcast(C)
    
    "Sampling"
    samples = fullClustering.map(lambda x : x if rand.random()<=min(t/sharedClusterSize.value[x[1]],1) else None).filter(lambda x: x!=None).cache()
    S_ = sorted(samples.map(lambda x: (x[1],x[0])).groupByKey().collect())
    S_ = [list(x[1]) for x in S_]
    
    "sample clusters size"
    C_s = [len(s) for s in S_]
    c_pr = [sum([dot(v,v) for v in vects]) for vects in S_]
    c_sums = [list(map(sum, zip(*vects))) for vects in S_]
    
    samples = samples.collect()
    
    clusteringSample = sc.broadcast(samples)
    
    start1 = time.time_ns()
    
    s = []
    for i in range(len(clusteringSample.value)):
        x = samples[i][0]
        y = samples[i][1]
        x_sq = dot(x,x)
        flag = 0
        if x in clusteringSample.value:
            flag = 1
        sums = [(-2*dot(x,c_sums[i]) + c_pr[i] + C_s[i]*x_sq)/(C_s[i]-flag) for i in range(len(C_s))]
        a = sums.pop(y)
        b = min(sums)
        s.append((b-a)/max(b,a))
    exactSilhSample = sum(s)/len(s)

    end1 = time.time_ns()
    
    start0 = time.time_ns()
    
    fullClustering = (fullClustering.map(lambda x: bigBrainMap(x,c_sums,c_pr, C_s, sharedClusterSize,t))
            .reduceByKey(add))
    approxSilhFull = float(fullClustering.collect()[0][1])/N
    
    end0 = time.time_ns()
    
    print("Value of approxSilhFull = %f"%(approxSilhFull))
    print("Time to compute approxSilhFull = %d ms"%(int((end0-start0)/1000000)))
    print("Value of exactSilhSample = %f"%(exactSilhSample))
    print("Time to compute exactSilhSample = %d ms"%(int((end1-start1)/1000000)))
    
if __name__ == "__main__":
    main()