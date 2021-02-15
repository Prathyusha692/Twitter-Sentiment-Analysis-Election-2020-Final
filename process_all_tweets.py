from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, Row, ArrayType, StringType
import pyspark.sql.functions as F
from pyspark.sql.functions import *
import json
import os
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
        return builtins.max(str.items(), key=lambda k: k[0], default=0)[1]
    else:
        return ''


def get_state_code(location):
    state_list=["AK","AL","AZ","AR","CA","CO","CT","DC","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"]
    state_abbr = {'alabama': 'AL','alaska': 'AK','american samoa': 'AS','arizona': 'AZ','arkansas': 'AR','california': 'CA','colorado': 'CO','connecticut': 'CT','delaware': 'DE','district of columbia': 'DC','florida': 'FL','georgia': 'GA','guam': 'GU','hawaii': 'HI','idaho': 'ID','illinois': 'IL','indiana': 'IN','iowa': 'IA','kansas': 'KS','kentucky': 'KY','louisiana': 'LA','maine': 'ME','maryland': 'MD','massachusetts': 'MA','michigan': 'MI','minnesota': 'MN','mississippi': 'MS','missouri': 'MO','montana': 'MT','nebraska': 'NE','nevada': 'NV','new hampshire': 'NH','new jersey': 'NJ','new mexico': 'NM','new york': 'NY','north carolina': 'NC','north dakota': 'ND','northern mariana islands':'MP','ohio': 'OH','oklahoma': 'OK','oregon': 'OR','pennsylvania': 'PA','puerto rico': 'PR','rhode island': 'RI','south carolina': 'SC','south dakota': 'SD','tennessee': 'TN','texas': 'TX','utah': 'UT','vermont': 'VT','virgin islands': 'VI','virginia': 'VA','washington': 'WA','west virginia': 'WV','wisconsin': 'WI','wyoming': 'WY'}
    loc_list=location.split(" ")
    loc_index=len(loc_list)-1
    state_code=loc_list[loc_index].upper()
    if(len(loc_list[loc_index])==2 and state_code in state_list ):
      my_dict = {'1': [state_code]}
      return my_dict
    else:
        try:
            state_code=state_abbr[loc_list[loc_index]]
            my_dict = {'1': [state_code]}
            return my_dict
        except:
          words = location.split()
          grouped_words = [' '.join(words[i: i + 2]) for i in range(0, len(words), 1)] + words
          word_list = sorted(list(dict.fromkeys(grouped_words)), key=len)
          states = []
          res = {}
          for i in broadcastStates[0]:
              for j in i:
                  for name in word_list:
                      if name in j['state_meta']:
                          resStr = j['state_code']
                          states.append(resStr)
              my_dict = {i:states.count(i) for i in states}
              for i, v in my_dict.items():
                res[v] = [i] if v not in res.keys() else res[v] + [i]
              if res:
                  return res
              else:
                  empty_res = {'1': ['No State']}
                  return empty_res

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
    
    #define UDFs
    geo_udf = udf(lambda x: get_state_code(x), StringType())
    geo_udf_1 = udf(lambda x: get_max_state_code(x), ArrayType(StringType()))
    
    #load geo true
    geo_true_df = spark.read.parquet(r"C:\Users\Prathyusha\Desktop\states_reports\data\geo_true")
    geo_true_df = geo_true_df.filter(geo_true_df.country_code == 'US')
    geo_true_df = geo_true_df.drop('_id', 'coordinates', 'country_code').withColumnRenamed("city_state", "location")
    geo_true_df=geo_true_df.where(col("location").isNotNull())
    geo_true_df=geo_true_df.select("id","location","text","time")
    
    #load geo false
    geo_false_df = spark.read.parquet(r"C:\Users\Prathyusha\Desktop\states_reports\data\geo_false")
    geo_false_df=geo_false_df.where(col("location").isNotNull())
    geo_false_df = geo_false_df.drop('_id', 'lang')
    
    #process tweets
    df = unionAll(geo_true_df, geo_false_df).distinct()
    df = df.withColumn("location_trans", F.lower(F.col("location")))
    df = df.withColumn('location_trans', regexp_replace('location_trans', '[^a-zA-Z0-9_]+', ' '))
    df=df.filter(df["location_trans"]!=" ")
    df = df.withColumn('location_dict', geo_udf('location_trans'))
    df = df.withColumn('location_arr', geo_udf_1('location_dict'))
    df.write.option("mode","overwrite").save(r"C:\Users\Prathyusha\Desktop\states_reports\final_output")

if __name__ == '__main__':
    main()