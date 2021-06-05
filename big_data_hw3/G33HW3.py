from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
import sys
import os
import random as rand
from operator import add
import time

def strToTuple(line):
    """Input parsing function"""
    ch = line.strip().split(" ")
    point = tuple(float(ch[i]) for i in range(len(ch)))
    return point

def bigBrainMap(elem, s_sums, s_squares, s_sizes):
    x = elem[0]
    y = elem[1]
    x_sq = dot(x,x)
    w = [(-2*dot(x,s_sums[i]) + s_squares[i] + 
          s_sizes[i]*x_sq)/s_sizes[i] for i in range(len(s_sizes))]
    a = w.pop(y)
    b = min(w)
    return [(b-a)/max(a,b)]

def dot(x,y):
    """simple dot product between two vectors"""
    return sum([x[i]*y[i] for i in range(len(x))])

def main():
    assert len(sys.argv) == 7, "Usage: python G33HW2.py <file_name> <kstart> <h> <iter> <M> <L>"

    conf = (SparkConf().setAppName('Homework3').set('spark.locality.wait','0s'))
    sc = SparkContext(conf=conf)

    data_path = sys.argv[1]
    #assert os.path.isfile(data_path), "File or folder not found"
    
    kstart = sys.argv[2]
    assert kstart.isdigit(), "kstart must be an integer"
    kstart = int(kstart)
    
    h = sys.argv[3]
    assert h.isdigit(), "h must be an integer"
    h = int(h)
    
    iter = sys.argv[4]
    assert iter.isdigit(), "iter must be an integer"
    iter = int(iter)
    
    M = sys.argv[5]
    assert M.isdigit(), "M must be an integer"
    M = int(M)
    
    L = sys.argv[6]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    t_read0 = time.time()
    inputPoints = sc.textFile(data_path, minPartitions=L).cache()
    inputPoints= inputPoints.map(strToTuple)
    inputPoints = inputPoints.repartition(numPartitions=L)
    delta_t_read = time.time() - t_read0
    print("Time for input reading = %d ms\n"%(int(delta_t_read*1000)))
    N = inputPoints.count()
    
    for i in range(kstart,kstart+h):
        t = M/i
        #Lloyds algorithm 
        start_cl = time.time()
        currentModel=sc.broadcast(KMeans.train(inputPoints,i,maxIterations=iter))
        currentClustering = inputPoints.map(lambda x: (x,currentModel.value.predict(x))).cache()
        end_cl = time.time()
        #Pointest P cluster sizes
        C = sorted(currentClustering.map(lambda x: x[1]).countByValue().items())
        
        C_final = [1 for _ in range(i)]
        for _ in C:
            C_final[_[0]] = _[1]
        sharedClusterSize = sc.broadcast(C_final)

        #Sampling
        samples = currentClustering.filter(lambda x: rand.random()<=min(t/sharedClusterSize.value[x[1]],1))
        S_ = sorted(samples.map(lambda x: (x[1],x[0])).groupByKey().collect())
        S_ = [list(x[1]) for x in S_]

        #sample clusters size
        C_s = [len(s) for s in S_]

        #Sum of the squares of every vector in a cluster for every cluster in S_
        s_sq = [sum([dot(v,v) for v in vects]) for vects in S_]

        #Sum of every vector in a cluster for every cluster in S_
        s_sums = [list(map(sum, zip(*vects))) for vects in S_]

        sampleClusterSize = sc.broadcast(C_s)#cluster size of sample
        sampleSquares = sc.broadcast(s_sq)#sum of squares for every sample clusterg gg
        sampleSums = sc.broadcast(s_sums)#sum of vector for every sample cluster

        start_mr = time.time()
        currentClustering = currentClustering.flatMap(lambda x: bigBrainMap(x,sampleSums.value,sampleSquares.value,sampleClusterSize.value)).cache()
        approxSilhFull = float(currentClustering.fold(0,add))/N
        end_mr = time.time()

        print("Number of clusters k = %d"%(i))
        print("Silhouette coefficient = %f"%(approxSilhFull))
        print("Time for clustering = %d ms"%(int((end_cl-start_cl)*1000)))
        print("Time for silhouette computation = %d ms\n"%(int((end_mr-start_mr)*1000)))
        
if __name__ == "__main__":
    main()