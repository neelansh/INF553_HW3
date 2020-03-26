import os
import sys
from pyspark import SparkContext, SparkConf
import json
import itertools
import math
import numpy as np
import time
from itertools import combinations
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error


appName = 'assignment3'
master = 'local[*]'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")


def read_test_data(test_path):
    def read_test_line(line):
        line = line.split(',')
        return (line[0].strip(), line[1].strip())
    
    rdd = sc.textFile(test_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).map(read_test_line)    
    return rdd


def read_train_data(train_path):
    def read_csv_line(line):
        line = line.split(',')
        return (line[0].strip(), line[1].strip(), line[2].strip())
    
    rdd = sc.textFile(train_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header).map(read_csv_line)
    return rdd

def save_output(output, path):
    output_file = open(path, 'wt')
    output_file.write('user_id, business_id, prediction\n')
    for line in output:
        output_file.write(line)
    output_file.close()
    return



if __name__ == '__main__':
    
    folder_path=sys.argv[1].strip()
    test_path=sys.argv[2].strip()
    output_path = sys.argv[3].strip()
    
    business_data = sc.textFile(os.path.join(folder_path, 'business.json')).map(json.loads).map(lambda x: (x['business_id'], x['review_count'], x['stars'])).collect()
    user_data = sc.textFile(os.path.join(folder_path, 'user.json')).map(json.loads).map(lambda x: (x['user_id'], x['review_count'], x['average_stars'])).collect()
    checkin_data = sc.textFile(os.path.join(folder_path, 'checkin.json')).map(json.loads).map(lambda x: (x['business_id'], sum(x['time'].values()))).collect()
    test_data = read_test_data(test_path).collect()
    train_data = read_train_data(os.path.join(folder_path, 'yelp_train.csv')).collect()
    business_data = pd.DataFrame(business_data, columns=['business_id', 'review_count', 'stars'])
    user_data = pd.DataFrame(user_data, columns=['user_id', 'review_count', 'average_stars'])
    checkin_data = pd.DataFrame(checkin_data, columns=['business_id', 'total_checkins'])
    train_data = pd.DataFrame(train_data, columns=['user_id', 'business_id', 'rating'])
    test_data = pd.DataFrame(test_data, columns=['user_id', 'business_id'])



    business_data = pd.merge(business_data, checkin_data, on='business_id', how='left')
    train_data = pd.merge(train_data, user_data, on='user_id', how='left')
    train_data = pd.merge(train_data, business_data, on='business_id', how='left')
    test_data = pd.merge(test_data, user_data, on='user_id', how='left')
    test_data = pd.merge(test_data, business_data, on='business_id', how='left')



    train_x = train_data.loc[:, ('review_count_x', 'average_stars', 'review_count_y', 'stars', 'total_checkins')].values
    train_y = train_data.loc[:, ('rating')].values
    test_x = test_data.loc[:, ('review_count_x', 'average_stars', 'review_count_y', 'stars', 'total_checkins')].values
#     test_y = test_data.loc[:, ('rating')].values



    model = xgb.XGBRegressor(objective ='reg:squarederror')
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    output = pd.concat((test_data.loc[:, ('user_id', 'business_id')], pd.DataFrame(pred, columns=['pred'])), axis=1)
    output.pred = output.pred.apply(lambda x: '5.0' if x>5 else str(x))
    output = output.apply(lambda row: ','.join(row)+'\n', axis=1).tolist()

    save_output(output, output_path)