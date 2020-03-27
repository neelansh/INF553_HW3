import os
import sys
from pyspark import SparkContext, SparkConf
import json
import itertools
import math
import numpy as np
import time
from itertools import combinations
import random


appName = 'assignment3'
master = 'local[*]'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")


def lsh(train_data_path):
    # Total number of users 11270
    p = 13591
    m = 11270
    number_of_hashes = 128
    random.seed(123)
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
        b = 64
        r = 2
        output = []
        for i in range(0, b):
            output.append(((i, hash(tuple(sig[r*i:(i+1)*r]))), doc_id))
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
    
    
    rdd = read_business(train_data_path)
    signatures = rdd.collect()
    index = index_signatures(signatures)
    business_sig = sc.parallelize(signatures).map(lambda x: (x[0], minhash(x[1], a, b, p, m)))
    candidates = business_sig.flatMap(hash_bands).groupByKey().map(lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) > 1).collect()
    reverse_index = {}
    candidates_index = {}

    for bid, c in candidates:
        candidates_index[bid] = c

    for bucket_id, bucket in candidates:
        for bid in bucket:
            reverse_index[bid] = bucket_id
            
    return candidates_index, reverse_index


def read_csv_line(line):
    line = line.split(',')
    return (line[0].strip(), line[1].strip(), line[2].strip())

def avg(item):
    vec = item[1]
    s = 0
    temp = {}
    for x in vec:
        s += float(x[1])
        temp[x[0]] = x[1]
        
    return (item[0], {'vec': temp, 'avg': s/len(vec)})

def read_business(data_path):
    rdd = sc.textFile(data_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).map(read_csv_line)
    
    business_users = rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().map(avg).collect()
    temp = {}
    for b, u in business_users:
        temp[b] = u
    return temp


def read_user(data_path):
    rdd = sc.textFile(data_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).map(read_csv_line)
    
    business_users = rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(avg).collect()
    temp = {}
    for u, b in business_users:
        temp[u] = b
    return temp


def pearson_correlation(business1, business2):
    vec_set1 = set()
    b1_sum = 0
    for user, rating in business1['vec'].items():
        b1_sum += float(rating)
        vec_set1.add(user)
    
    b2_sum = 0
    vec_set2 = set()
    for user, rating in business2['vec'].items():
        b2_sum += float(rating)
        vec_set2.add(user)
            
    vec_set = vec_set1.union(vec_set2) 
    if(len(vec_set) == 0):
        return 0.0
    b1_sum += (len(vec_set)-len(business1['vec'])) * 3.5
    b2_sum += (len(vec_set)-len(business2['vec'])) * 3.5
    b1_avg = b1_sum/len(vec_set)
    b2_avg = b2_sum/len(vec_set)

    corr_numerator = 0
    corr_denominator_p1 = 0
    corr_denominator_p2 = 0
    
    for user_id in vec_set:
        v1 = None
        if(user_id in business1['vec']):
            v1 = float(business1['vec'][user_id])-b1_avg
        else:
            v1 = business1['avg']-b1_avg
            
            
        if(user_id in business2['vec']):
            v2 = float(business2['vec'][user_id])-b2_avg
        else:
            v2 = business2['avg']-b2_avg
        
        corr_numerator += v1*v2
        corr_denominator_p1 += math.pow(v1, 2)
        corr_denominator_p2 += math.pow(v2, 2)
    
    denominator = math.pow(corr_denominator_p1, 0.5) * math.pow(corr_denominator_p2, 0.5)
    
    if(denominator != 0):
        return corr_numerator / denominator
    else:
        return 0.0



def calculate_rating(train_data, user_avg, reverse_index, candidates_index, business_id, user_id):
    if(not business_id in train_data and not user_id in user_avg):
        return "{},{},{}\n".format(user_id, business_id, 4.0)
    if(not business_id in train_data and user_id in user_avg):
        return "{},{},{}\n".format(user_id, business_id, user_avg[user_id]['avg'])
    if(business_id in train_data and not user_id in user_avg):
        return "{},{},{}\n".format(user_id, business_id, train_data[business_id]['avg'])
    
    business_data = train_data[business_id]
    similar_items = []
    for bid in candidates_index[reverse_index['FaHADZARwnY4yvlvpnsfGA']]:
        b = train_data[bid]
        if(not user_id in b['vec']):
            continue
        sim = pearson_correlation(business_data, b)
        if(sim > 0.0):
            similar_items.append((sim, float(b['vec'][user_id])))
#     similar_item = sorted(similar_items, key=lambda x: x[0], reverse=True)
    numerator = 0
    denominator = 0
    
    for similarity, rating in similar_items:
        weight = similarity# * math.pow(abs(similarity), rho-1)
        numerator += weight*rating
        denominator += weight
        
    if(denominator != 0):
        return "{},{},{}\n".format(user_id, business_id, numerator/denominator)
    else:
        return "{},{},{}\n".format(user_id, business_id, train_data[business_id]['avg'])

    
    
# def calculate_rating_partition(train_data, user_avg, test_data):
    
#     for (business_id, user_id) in test_data:
    
#         if(not business_id in train_data and not user_id in user_avg):
#             yield "{},{},{}\n".format(user_id, business_id, 4.0)
#             continue
#         if(not business_id in train_data and user_id in user_avg):
#             yield "{},{},{}\n".format(user_id, business_id, user_avg[user_id]['avg'])
#             continue
#         if(business_id in train_data and not user_id in user_avg):
#             yield "{},{},{}\n".format(user_id, business_id, train_data[business_id]['avg'])
#             continue

#         business_data = train_data[business_id]
#         similar_items = []
#         for bid, rating in user_avg[user_id]['vec'].items():
#             b = train_data[bid]
#             sim = pearson_correlation(business_data, b)
#             if(sim > 0.0):
#                 similar_items.append((sim, float(b['vec'][user_id])))
#     #     similar_item = sorted(similar_items, key=lambda x: x[0], reverse=True)
#         numerator = 0
#         denominator = 0

#         for similarity, rating in similar_items:
#             weight = similarity# * math.pow(abs(similarity), rho-1)
#             numerator += weight*rating
#             denominator += weight

#         if(denominator != 0):
#             yield "{},{},{}\n".format(user_id, business_id, numerator/denominator)
            
#         else:
#             yield "{},{},{}\n".format(user_id, business_id, train_data[business_id]['avg'])
            
    
    
def save_output(output, path):
    output_file = open(path, 'wt')
    output_file.write('user_id, business_id, prediction\n')
    for line in output:
        output_file.write(line)
    output_file.close()
    return
    
def read_test_data(test_path):
    def read_test_line(line):
        line = line.split(',')
        return (line[0].strip(), line[1].strip())
    
    rdd = sc.textFile(test_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).map(read_test_line).map(lambda x: (x[1], x[0]))#.join(train_data).map(lambda x: (x[0], x[1][0], x[1][1]))
    
    return rdd.collect()



def rmse(test_path, pred_path):
    test_data = {}
    count = 0
    with open(test_path, 'rt') as file:
        line = file.readline().strip()
        line = file.readline().strip()
        while(line):
            count+=1
            test_data[tuple(line.split(',')[:2])] = float(line.split(',')[2])
            line = file.readline().strip()
            
    sum_error_square = 0
    with open(pred_path, 'rt') as file:
        line = file.readline().strip()
        line = file.readline().strip()
        while(line):
            if(line == 'Null'):
                line = file.readline().strip()
                continue
            sum_error_square+=math.pow((test_data[tuple(line.split(',')[:2])]-float(line.split(',')[2])), 2)
            line = file.readline().strip()
            
    return math.pow(sum_error_square/count, 0.5)


# def expand(train_data, user_avg, user_id, business_id):
#     if(not user_id in user_avg or not business_id in train_data):
#         return []
#     output = []
#     for bid, r in user_avg[user_id]['vec'].items():
#         output.append(((business_id, user_id), (bid, r)))
        
#     return output

# def pearson_wrapper(train_data, bid1, bid2, bucket_index, candidate_index):
#     return pearson_correlation(train_data[bid1], train_data[bid2])



if __name__ == '__main__':
    
    st = time.time()
    
    train_path = sys.argv[1].strip()
    test_path = sys.argv[2].strip()
    output_path = sys.argv[3].strip()
    
    candidates_index, reverse_index = lsh(train_path)
    
    train_data = read_business(train_path)
    test_data = read_test_data(test_path)
    user_avg = read_user(train_path)
    
    output = sc.parallelize(test_data).map(lambda x: calculate_rating(train_data, user_avg, reverse_index, candidates_index, x[0], x[1])).collect()
    print(time.time()-st)
    save_output(output, output_path)
    print(rmse(test_path, output_path))