'''
To read the input text file (e.g., inputPath) containing a clustering
into the RDD full clustering do:

fullClustering = sc.textFile(inputPath).map(strToTuple)
'''

def strToTuple(line):
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1])) # returns (point, cluster_index)
