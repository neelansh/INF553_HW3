import os
import sys
from pyspark import SparkContext, SparkConf
import json
import itertools
import math
import random
import time
from itertools import combinations

appName = 'assignment3'
master = 'local[*]'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# Total number of users 11270
p = 13591
m = 64
number_of_hashes = 64
a = random.sample(range(1, p), number_of_hashes)
b = random.sample(range(0, p), number_of_hashes)


def read_csv_line(line):
    line = line.split(',')
    return (line[0].strip(), line[1].strip(), line[2].strip())

def read_business(data_path):
    rdd = sc.textFile(data_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).map(read_csv_line)
    unique_users = rdd.map(lambda x: x[0]).distinct().collect()    
    userid_index = {}
    for i, uid in enumerate(unique_users):
        userid_index[uid] = i
    
    business_users = rdd.map(lambda x: (x[1], userid_index[x[0]])).groupByKey().map(lambda x: (x[0], list(x[1])))
    
    return business_users

def calculate_hash(a, b, p, m, vec):
    return min([((a*x)+b)%p%m for x in vec])

def minhash(vector, a, b, p, m):
    return [calculate_hash(ai, bi, p, m, vector) for ai, bi in zip(a, b)]
#     return list(np.min((a.reshape(1, a.shape[0]) * vector.reshape(vector.shape[0], 1) + b) %p %m, axis=0))

def hash_bands(x):
    doc_id, sig = x
    b = 16
    r = 4
    output = []
    for i in range(0, b):
        output.append(((i, hash(frozenset(sig[r*i:(i+1)*r]))), doc_id))
    return output


def jaccard_sim(x, y):
    x = set(x)
    y = set(y)
    return len(x.intersection(y)) / len(x.union(y))


def index_signatures(signatures):
    index = {}
    for x, y in signatures:
        index[x] = y
        
    return index

def find_similarity(candidate, index):
    bucket = candidate[1]
    output = []
    for candidate1, candidate2 in combinations(bucket, 2):
        sim = jaccard_sim(index[candidate1], index[candidate2])
        if(sim >= 0.5):
            pair_sorted = sorted([candidate1, candidate2])
            output.append((pair_sorted[0], pair_sorted[1], sim))
            
    return output

def save_output(output, file_path):
    output = sorted(output)
    with open(file_path, "wt") as f:
        f.write('business_id_1, business_id_2, similarity\n')
        for pairs in output:
            f.write(pairs)
    return 

if __name__ == '__main__':
    rdd = read_business(sys.argv[1].strip())
    signatures = rdd.collect()
    index = index_signatures(signatures)
    business_sig = sc.parallelize(signatures).map(lambda x: (x[0], minhash(x[1], a, b, p, m)))
    candidates = business_sig.flatMap(hash_bands).groupByKey().map(lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) > 1)
    similar_pairs = candidates.flatMap(lambda x: find_similarity(x, index)).map(lambda pairs: ','.join([str(x) for x in pairs])+'\n').distinct().collect()
    save_output(similar_pairs, sys.argv[2].strip())