from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import time

def strToTuple(line):
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1]))    
    

def main():
    print(len(sys.argv))
    assert len(sys.argv) == 2, "Usage: python G33HW1.py <file_name>"
    
    conf = SparkConf().setAppName('G33HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    RawData = sc.textFile(data_path).cache()
    RawData = RawData.flatMap(strToTuple)
    v = sorted(RawData.countByValue().items()) 
    print(v)
    v = v[:,1]
    
    sharedClusterSize = sc.broadcast(v)

if __name__ == "main":
    main()