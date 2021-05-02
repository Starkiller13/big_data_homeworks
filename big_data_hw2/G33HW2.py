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
    assert len(sys.argv) == 2, "Usage: python G33HW1.py <file_name>"
    
    conf = SparkConf().setAppName('G33HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    RawData = sc.textFile(data_path).cache()
    RawData = RawData.map(strToTuple)
    
    C = sorted(RawData.map(lambda x: x[1]).countByValue().items()) 
    C = [C[i][1] for i in range(len(C))]
    
    sharedClusterSize = sc.broadcast(C)
    

if __name__ == "__main__":
    main()