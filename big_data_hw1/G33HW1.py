from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand

def map1(row):
    rr = row.split(',')
    return (rr[0],1)

def map2(data):
    return
def doit(data, K,T):
    rnorm = (data.flatMap(map1)
            .reduceByKey(lambda a,b: (a[0]+b[0]))) 
    return rnorm
    
    
def main():
    assert len(sys.argv) == 4, "Usage: python G33HW1.py <K> <T> <file_name>"
    
    conf = SparkConf().setAppName('G33HW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)
    
    T = sys.argv[2]
    assert T.isdigit(), "T must be an integer"
    T = int(K)
    
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    RawData = sc.textFile(data_path,minPartitions=K).cache()
    RawData.repartition(numPartitions=K)
    normData = doit(RawData,K,T)
    print(normData.collect())


if __name__ == "__main__":
	main()