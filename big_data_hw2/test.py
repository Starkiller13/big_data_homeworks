import sys
import os
import random as rand
from pyspark import SparkContext, SparkConf

def strToTuple(line):
    """Input parsing function"""
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1]))    

def bigBrainMap(elem, c_sums, c_prods, s_sizes, c_sizes, t):
    x = elem[0]
    y = elem[1]
    x_sq = float(dot(x,x))
    w = [(s_sizes[i]*x_sq - 2*float(dot(x,c_sums[i])) + c_prods[i])/min(t,c_sizes.value[i]) for i in range(len(c_sizes.value))]
    w[y]= ((s_sizes[y]-1)*x_sq - 2*dot(x,c_sums[y]) + c_prods[y])/min(t,c_sizes.value[y])
    a = w[y]
    w.pop(y)
    b = min(w)
    return (0,(b-a)/max(b,a))

conf = SparkConf().setAppName('test.py').setMaster("local[*]")
conf.set("spark.kryo.registrationRequired", "true")
sc = SparkContext(conf=conf)
data_path = sys.argv[1]

t=3 
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

c_sq = [float(sum([dot(v,v) for v in vects])) for vects in S_]
c_sums = [[float(sum(x)) for x in zip(*vects)] for vects in S_]
samples = samples.collect()

clusteringSample = sc.broadcast(samples)

start1 = time.time_ns()

s = []
for i in range(len(clusteringSample.value)):
    x = samples[i][0]
    y = samples[i][1]
    c_sum = sum([dist(x,p) for p in S_[y]])
    l_sums = [sum([dist(x,p) for p in S_[j]])/C_s[j] for j in range(0,k) if j!=y]
    a = c_sum/(C_s[y]-1)
    b = min(l_sums)
    s.append((b-a)/max(b,a))
    
for i in range(len(clusteringSample.value)):
    x = samples[i][0]
    y = samples[i][1]
    x_sq = float(dot(x,x))
    w = [(s_sizes[i]*x_sq - 2*float(dot(x,c_sums[i])) + c_prods[i])/min(t,c_sizes.value[i]) for i in range(len(c_sizes.value))]
    w[y]= ((s_sizes[y]-1)*x_sq - 2*dot(x,c_sums[y]) + c_prods[y])/min(t,c_sizes.value[y])
    a = w[y]
    w.pop(y)
    b = min(w)
    
exactSilhSample = sum(s)/len(s)