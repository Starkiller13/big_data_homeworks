from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand


def word_count_per_doc(document, K=-1):
		pairs_dict = {}
		for word in document.split(' '):
			if word not in pairs_dict.keys():
				pairs_dict[word] = 1
			else:
				pairs_dict[word] += 1
		if K == -1:
			return [(key, pairs_dict[key]) for key in pairs_dict.keys()]
		else:
			return [(rand.randint(0,K-1),(key, pairs_dict[key])) for key in pairs_dict.keys()]

def gather_pairs(pairs):
	pairs_dict = {}
	for p in pairs[1]:
		word, occurrences = p[0], p[1]
		if word not in pairs_dict.keys():
			pairs_dict[word] = occurrences
		else:
			pairs_dict[word] += occurrences
	return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def word_count_1(docs):
	word_count = (docs.flatMap(word_count_per_doc) # <-- MAP PHASE (R1)
				 .reduceByKey(lambda x, y: x + y)) # <-- REDUCE PHASE (R1)
	return word_count

def word_count_2(docs, K):
	word_count = (docs.flatMap(lambda x: word_count_per_doc(x, K)) # <-- MAP PHASE (R1)
				 .groupByKey()                            # <-- REDUCE PHASE (R1)
				 .flatMap(gather_pairs)                   
				 .reduceByKey(lambda x, y: x + y))        # <-- REDUCE PHASE (R2)
	return word_count
	
def word_count_3(docs, K):
	word_count = (docs.flatMap(word_count_per_doc) # <-- MAP PHASE (R1)
				 .groupBy(lambda x: (rand.randint(0,K-1))) # <-- REDUCE PHASE (R1)
				 .flatMap(gather_pairs)                   
				 .reduceByKey(lambda x, y: x + y))        # <-- REDUCE PHASE (R2)
	return word_count

def word_count_2_with_partition(docs):
	def gather_pairs_partitions(pairs):
		pairs_dict = {}
		for p in pairs:
			word, occurrences = p[0], p[1]
			if word not in pairs_dict.keys():
				pairs_dict[word] = occurrences
			else:
				pairs_dict[word] += occurrences
		return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

	word_count = (docs.flatMap(word_count_per_doc) # <-- MAP PHASE (R1)
		.mapPartitions(gather_pairs_partitions)    # <-- REDUCE PHASE (R1)
		.groupByKey()                              # <-- REDUCE PHASE (R2)
		.mapValues(lambda vals: sum(vals)))

	return word_count

def main():

	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 3, "Usage: python WordCountExample.py <K> <file_name>"

	# SPARK SETUP
	conf = SparkConf().setAppName('WordCountExample').setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# INPUT READING

	# 1. Read number of partitions
	K = sys.argv[1]
	assert K.isdigit(), "K must be an integer"
	K = int(K)

	# 2. Read input file and subdivide it into K random partitions
	data_path = sys.argv[2]
	assert os.path.isfile(data_path), "File or folder not found"
	docs = sc.textFile(data_path,minPartitions=K).cache()
	docs.repartition(numPartitions=K)

	# SETTING GLOBAL VARIABLES
	numdocs = docs.count()
	print("Number of documents = ", numdocs)

	# STANDARD WORD COUNT with reduceByKey
	print("Number of distinct words in the documents using reduceByKey =", word_count_1(docs).count())

	# IMPROVED WORD COUNT with groupByKey
	print("Number of distinct words in the documents using groupByKey =", word_count_2(docs, K).count())

	# IMPROVED WORD COUNT with groupBy
	print("Number of distinct words in the documents using groupBy =", word_count_3(docs, K).count())

	# IMPROVED WORD COUNT with mapPartitions
	wordcount = word_count_2_with_partition(docs)
	numwords = wordcount.count()
	print("Number of distinct words in the documents using mapPartitions =", numwords)

	# COMPUTE AVERAGE WORD LENGTH
	average_word_len = wordcount.keys().map(lambda x: len(x)).reduce(lambda x,y: x+y)
	print("Average word length = ", average_word_len/numwords)


if __name__ == "__main__":
	main()
