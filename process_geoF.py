#from __future__ import (absolute_import, division, print_function, unicode_literals)

from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql import DataFrame
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, Row, ArrayType, StringType
import pyspark.sql.functions as F
from pyspark.sql.functions import *
import json
import os
#import __builtin__
from operator import add


broadcastStates=[]

def init_spark():
    spark = SparkSession.builder.appName("SparkTest") \
        .config("spark.memory.fraction", 0.8) \
        .config("spark.sql.shuffle.partitions", "800") \
        .config("spark.sql.debug.maxToStringFields", 1000) \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.parquet.binaryAsString", "true") \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc

## Added UDf to Get max_state_code
def get_max_state_code(str):

    import builtins
    if builtins.max(str.items(), key = lambda k : k[0], default=0)[0] in str.keys():
        # print(builtins.max(str.items(), key=lambda k : k[0], default=0)[0])
        # print(builtins.max(str.items(), key=lambda k: k[0], default=0)[1])
        return builtins.max(str.items(), key=lambda k: k[0], default=0)[1]
    else:
        return ''


def get_state_code(location):
    state_list=["AK","AL","AZ","AR","CA","CO","CT","DC","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]
    loc_list=location.split(" ")
    loc_index=len(loc_list)-1
    state_code=loc_list[loc_index].upper()
    if(len(loc_list[loc_index])==2 and state_code in state_list ):
      my_dict = {'1': [state_code]}
      return my_dict
    else:
      words = location.split()
      grouped_words = [' '.join(words[i: i + 2]) for i in range(0, len(words), 1)] + words
      word_list = sorted(list(dict.fromkeys(grouped_words)), key=len)
      # print(word_list)
      states = []
      res = {}
      for i in broadcastStates[0]:
          for j in i:
              for name in word_list:
                  if name in j['state_meta']:
                      resStr = j['state_code']
                      # print(resStr)
                      states.append(resStr)
          my_dict = {i:states.count(i) for i in states}
          for i, v in my_dict.items():
            res[v] = [i] if v not in res.keys() else res[v] + [i]
          if res:
              return res
          else:
              empty_res = {'1': ['No State']}
              return empty_res
          # return res

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

def get_dir(path):
    file_paths = []
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            file_paths.append(os.path.join(root, name))
    return file_paths

def main():
    spark, sc = init_spark()
    ps = sc.wholeTextFiles(r"C:\Users\Prathyusha\Desktop\states_reports\us_state_meta.json").values().map(json.loads)
    broadcastStates.append(spark.sparkContext.broadcast(ps.map(lambda x: x).collect()).value)
    geo_false_df = spark.read.parquet(r"C:\Users\Prathyusha\Desktop\states_reports\data\geo_false")
    geo_false_df=geo_false_df.where(col("location").isNotNull())
    #geo_false_df = geo_false_df.filter(geo_false_df.country_code == 'US')
    geo_false_df = geo_false_df.withColumn("location_trans", F.lower(F.col("location")))
    geo_false_df = geo_false_df.withColumn('location_trans', regexp_replace('location_trans', '[^a-zA-Z0-9_]+', ' '))
# geo_false_df.select("location","location_dict").show(truncate=False,n=500)
    geo_false_df=geo_false_df.filter(geo_false_df["location_trans"]!=" ")#17132
#After removing location_dict with " " -- 598017
    geo_udf = udf(lambda x: get_state_code(x), StringType())
    geo_false_df = geo_false_df.withColumn('location_dict', geo_udf('location_trans'))#.limit(1000)
# geo_false_df.filter(geo_false_df["location_arr"]=="{1=[No State]}").count()
## Added To get Max Key,value of State Codes
    geo_udf_1 = udf(lambda x: get_max_state_code(x), ArrayType(StringType()))
    geo_false_df = geo_false_df.withColumn('location_arr', geo_udf_1('location_dict'))
# df=geo_false_df.groupBy("location_arr").count()
# df.show(truncate=False,n=500)
# {1=[No State]} 
#geo_false_df.select("location_trans","location_dict","location_arr").show(truncate=False,n=5000)
    df = geo_false_df.withColumn("location_array",col("location_arr").cast("string"))
    df = df.drop("location_arr")    
    df.show(10)
    df.printSchema()
    #df=df.select("id","time","location","location_trans","location_array")
    #df.coalesce(10).write.format('com.databricks.spark.csv').save(r"C:\Users\Prathyusha\Desktop\states_reports\output",header = 'true')   
if __name__ == '__main__':
    main()