from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from operator import add
import time


def strToTuple(line):
    """Input parsing function"""
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1]))    

def bigBrainMap(elem, s_sums, s_squares, s_sizes, c_sizes, t):
    """ ---Sometimes midnight ideas are great---
        Main idea: decomposition of the overall distance per cluster
        given a fixed x and a cluster S with y in S
        sum(L2_squared_distances) = sum_over_y_in_S(dot(x-y,x-y)) = 
            = sum_over_y_in_S(dot(x,x)+dot(y,y)-2*dot(x,y)) =
            = |S|*dot(x,x) + sum_over_y_in_S(dot(y,y)) - 2 * dot(x,sum_over_y_in_S(y))
        
        where sum_over_y_in_S(dot(y,y)) = sum_of_sqares and sum_over_y_in_S(y) = sum_of_y
        
        but this two quantities can be computed once for every sample cluster S_i so 
        to compute the entire sum I can just compute on the fly dot(x,x) and 
        dot(x,sum_of_y) (that are O(1) operations versus the O(|S|) ops needed to compute all 
        the distances)
    """
    x = elem[0]
    y = elem[1]
    x_sq = dot(x,x)
    w = [(-2*dot(x,s_sums.value[i]) + s_squares.value[i] + 
          s_sizes.value[i]*x_sq)/s_sizes.value[i] for i in range(len(c_sizes.value))]
    a = w.pop(y)
    b = min(w)
    return (0,(b-a)/max(a,b))

def dot(x,y):
    """simple dot product between two vectors"""
    return sum([x[i]*y[i] for i in range(len(x))])

def main():
    assert len(sys.argv) == 4, "Usage: python G33HW2.py <file_name> <k> <t>"
    
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

    #Pointest P cluster sizes
    C = sorted(fullClustering.map(lambda x: x[1]).countByValue().items())
    C = [C[i][1] for i in range(len(C))]
    
    sharedClusterSize = sc.broadcast(C)
    
    #Sampling
    samples = fullClustering.map(lambda x : x if rand.random()<=min(t/sharedClusterSize.value[x[1]],1) else None).filter(lambda x: x!=None).cache()
    S_ = sorted(samples.map(lambda x: (x[1],x[0])).groupByKey().collect())
    S_ = [list(x[1]) for x in S_]
    
    #sample clusters size
    C_s = [len(s) for s in S_]
    
    #Sum of the squares of every vector in a cluster for every cluster in S_
    s_sq = [sum([dot(v,v) for v in vects]) for vects in S_]
    
    #Sum of every vector in a cluster for every cluster in S_
    s_sums = [list(map(sum, zip(*vects))) for vects in S_]
    
    samples = samples.collect()
    
    clusteringSample = sc.broadcast(samples)
    sampleClusterSize = sc.broadcast(C_s)#cluster size of sample
    sampleSquares = sc.broadcast(s_sq)#sum of squares for every sample clusterg gg
    sampleSums = sc.broadcast(s_sums)#sum of vector for every sample cluster
    
    #Sequential exact silhouette coefficient for clusteringSample
    start_seq = time.time_ns()
    
    s = []
    for i in range(len(clusteringSample.value)):
        x = samples[i][0]
        y = samples[i][1]
        x_sq = dot(x,x)
        flag = 0
        if x in clusteringSample.value:
            flag = 1
        sums = [(-2*dot(x,s_sums[i]) + s_sq[i] + C_s[i]*x_sq)/(C_s[i]-flag) for i in range(len(C_s))]
        a = sums.pop(y)
        b = min(sums)
        s.append((b-a)/max(b,a))
    exactSilhSample = float(sum(s)/len(s))

    end_seq = time.time_ns()
    
    #MR approach to compute the approximate silhouette coefficient for fullClustering
    start_mr = time.time_ns()
    
    fullClustering = (fullClustering.map(lambda x: bigBrainMap(x,sampleSums,sampleSquares,sampleClusterSize,sharedClusterSize,t)).cache()
            .reduceByKey(add))
    approxSilhFull = float(fullClustering.collect()[0][1])/N
    
    end_mr = time.time_ns()
    
    print("Value of approxSilhFull = %f"%(approxSilhFull))
    print("Time to compute approxSilhFull = %d ms"%(int((end_mr-start_mr)/1000000)))
    print("Value of exactSilhSample = %f"%(exactSilhSample))
    print("Time to compute exactSilhSample = %d ms"%(int((end_seq-start_seq)/1000000)))
    
if __name__ == "__main__":
    main()