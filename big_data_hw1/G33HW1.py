from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand

def map1(row):
    rr = row.split(",")
    return [(rr[1],(rr[0],float(rr[2])))]

def map2(data): 
    pairs_list = []
    sum_rat = 0
    sum_rec = 0
    for p in data[1]:
        rating =  p[1]    
        sum_rat += rating
        sum_rec += 1
    avg = sum_rat/sum_rec
    for p in data[1]:
        product , rating = p[0] , p[1]
        pairs_list.append([product,rating-avg])

    return pairs_list

def map3(data, k):
    return [(rand.randint(0,k-1),(data[0],data[1]))]

def map4(data):
    pairs_dict={}
    for p in data[1]:
        prod, nr = p[0], p[1]
        if prod not in pairs_dict.keys():
            pairs_dict[prod] = nr
        else:
            if nr > pairs_dict[prod]:
                pairs_dict[prod] = nr
    return [(key, (key, pairs_dict[key])) for key in pairs_dict.keys()]

def map5(data):
    pairs_dict={}
    for p in data[1]:
        prod, nr = p[0], p[1]
        if prod not in pairs_dict.keys():
            pairs_dict[prod] = nr
        else:
            if nr > pairs_dict[prod]:
                pairs_dict[prod] = nr
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def doit(data, K):
    rnorm = (data.flatMap(map1)#<- MAP PHASE R1
            .groupByKey()# <- REDUCE PHASE R1
            .flatMap(map2)) 
    return rnorm
    
def doit2(data,K):
    maxnorm = (data.flatMap(lambda x: map3(x,K)) #<- MAP PHASE R1
                .groupByKey() # <- REDUCE PHASE R1
                .flatMap(map4)# <- MAP PHASE R2
                .groupByKey()# <- REDECE PHASE R2
                .flatMap(map5)
                )
    return maxnorm

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
    normalizedRatings = doit(RawData,K) #NORM RATINGS CALC
    maxNormRatings = doit2(normalizedRatings,K) # MAX NORM RATING CALC


if __name__ == "__main__":
	main()