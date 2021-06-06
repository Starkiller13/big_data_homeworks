# Bid Data Homeworks (unipd 2020/2021)

Made by Andrea Costalonga, Elena Bettella, Amanpreet Singh

# Prerequisites
You need to have java and spark installed on your machine.
Pyspark is also needed
```bash
pip install pyspark
```

# Homework 1: User Ratings
Basis of Map Reduce paradigm, our goal was to fetch and make some 
operations over a online shopping dataset of reviews.
Original instructions can be found [here](hw1_description)

Usage(starting from the main directory)
```bash
cd /big_data_hw1
python G33HW1.py <K> <T> <file_name>
```

Mark: 7.0/7.0

# Homework 2: Clustering
Starting from an already clustered pointset we had to compute an 
approximate silhouette coefficient for the entire pointset and 
the exact silhouette coefficient for a sample obtained using poisson sampling.
Original instructions can be found [here](hw2_description)

Usage(starting from the main directory)
```bash
cd /big_data_hw2
# k number of clusters in which the points are classified, 
# t parameter used for the poisson sampling
python G33HW2.py <file_name> <k> <t>
```

Mark: 7.0/7.0

# Homework 3: KMeans and Silhouette
Taking as starting point the program coded in HW2 we had to implement a script
in which we had to estimate the value of the silhouette of a clustering for multiple value of k.
The program takes in input a *unclustered pointset*, a starting value *kstart*, *h* that defines 
the endpoint of the loop, *iter* as the number of kmeans iterations, *M* used to calculate the
value *t* in each iteration and *L* that defines the number of partition in which the file will 
be split. 
This time we had to use the CluodVeneto infrastructure to execute the computations; we logged in the 
server through ssh via another ssh connection to the local login server of the departement (DEI).
More infos on the experiments made can be found [here](hw3_description)

Normal usage(no CloudVeneto):
```bash
cd /big_data_hw3
python G33HW3.py <file_name> <kstart> <h> <iter> <M> <L>
```

To run the program in CloudVeneto:
```bash
spark-submit --conf spark.pyspark.python=python3 --num-executors <number_of_executors> G33HW3.py <file_name> <kstart> <h> <iter> <M> <L>
```
