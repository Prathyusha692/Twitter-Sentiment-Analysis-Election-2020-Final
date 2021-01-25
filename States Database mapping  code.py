import sparkpickle
from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql import DataFrame
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, Row, ArrayType, StringType
import pyspark.sql.functions as F
from pyspark.sql.functions import *
import json

broadcastStates=[]

def init_spark():
    spark = SparkSession.builder.appName("Twitter Analysis") \
        .config("spark.memory.fraction", 0.8) \
        .config("spark.sql.shuffle.partitions", "800") \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc

def get_new_us_code(str):
    words = str.split()
    grouped_words = [' '.join(words[i: i + 2]) for i in range(0, len(words), 1)] + \
                    [' '.join(words[i: i + 1]) for i in range(0, len(words), 1)]
    word_list = sorted(list(dict.fromkeys(grouped_words)), key=len)
    # print(word_list)

    for i in broadcastStates[0]:
        for j in i:
            # print(j['country_code'])
            # print(j['comments'])
            for name in word_list:
                if name in j['comments']:
                    resStr = j['country_code']
                    # print(resStr)
                    return resStr

    return None

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)


def main():
    spark, sc = init_spark()

    # Read US.Metadata Json File
    ps = sc.wholeTextFiles(r"reference\us_state_meta_latest.json").values().map(json.loads)
    broadcastStates.append(spark.sparkContext.broadcast(ps.map(lambda x: x).collect()).value)
    # print(broadcastStates)
    # print(ps.map(lambda x: x).collect())
    # for i in broadcastStates:
    #    for j in i:
    #        print(j)

    # Read Geo True Json file
    geo_true_df = spark.read.json("data\geo_true.json")
    # print(geo_true_df.printSchema())
    # print(geo_true_df.show(truncate=False))

    # Read Geo True Json file
    # geo_false_df = spark.read.json("data\geo_false.json") #.repartition(100)
    # print(geo_false_df.printSchema())
    # print(geo_false_df.show(truncate=False))

    geo_true_df = geo_true_df.filter(geo_true_df.country_code == 'US')
    geo_true_df = geo_true_df.drop('_id', 'coordinates', 'country_code').withColumnRenamed("city_state", "location")
    geo_true_df = geo_true_df.withColumn("new_location", F.lower(F.col("location")))
    geo_true_df = geo_true_df.withColumn('new_location', regexp_replace('new_location', '^[a-zA-Z\']+', ' '))
    # print(geo_true_df.show(truncate=False))
    # print(geo_true_df.count())

    # Register UDF
    geo_udf = udf(lambda x: get_new_us_code(x), StringType())
    geo_true_df = geo_true_df.withColumn('new_location', geo_udf('new_location'))
    # print(geo_true_df.show(truncate=False))

    """geo_false_df = geo_false_df.drop('lang', '_id')
    print(geo_false_df.show(truncate=False))
    print(geo_false_df.count())
    geo_false_df = geo_false_df.withColumn("new_location", F.lower(F.col("location")))
    geo_false_df = geo_false_df.withColumn('new_location', regexp_replace('new_location', '^[a-zA-Z\']+', ' '))
    geo_false_df = geo_false_df.withColumn('new_location', geo_udf('new_location'))
    print(geo_false_df.show(truncate=False))
    print(geo_false_df.count())
    
    df = unionAll(geo_true_df, geo_false_df).distinct().show()
    print(df.show(truncate=False))"""

    # tweets = df
    tweets = geo_true_df

    # print("Total Tweets \t\t\t\t: ", tweets.count())

    # joeBiden tweets excluding trump
    joe_only = tweets.filter(
        (tweets['text'].rlike("[Jj]oe|[Bb]iden") == True) & (tweets['text'].rlike("[Dd]onald|[Tt]rump") == False))
    # print("Only Joe Biden Tweets \t\t\t: ", joe_only.count())

    trump_only = tweets.filter(
        (tweets['text'].rlike("[Jj]oe|[Bb]iden") == False) & (tweets['text'].rlike("[Dd]onald|[Tt]rump") == True))
    # print("Only Donald Trump Tweets \t\t: ", trump_only.count())

    joe_and_trump = tweets.filter(
        (tweets['text'].rlike("[Dd]onald|[Tt]rump")) & (tweets['text'].rlike("[Jj]oe|[Bb]iden")))
    # print("Both Joe_Biden & Trump Tweets \t\t: ", joe_and_trump.count())

    not_joe_trump = tweets.filter(
        ~(tweets['text'].rlike("[Dd]onald|[Tt]rump")) & ~(tweets['text'].rlike("[Jj]oe|[Bb]iden")))
    # print("Tweets without Joe_Biden & Trump \t: ", not_joe_trump.count())

    sid = SentimentIntensityAnalyzer()

    udf_priority_score = udf(lambda x: sid.polarity_scores(x), returnType=FloatType())  # Define UDF function
    udf_compound_score = udf(lambda score_dict: score_dict['compound'])
    udf_comp_score = udf(lambda c: 'pos' if c >= 0.05 else ('neu' if (c > -0.05 and c < 0.05) else 'neg'))

    trump_only = trump_only.withColumn('scores', udf_priority_score(trump_only['text']))
    trump_only = trump_only.withColumn('compound', udf_compound_score(trump_only['scores']))
    trump_only = trump_only.withColumn('comp_score', udf_comp_score(trump_only['compound']))

    joe_only = joe_only.withColumn('scores', udf_priority_score(joe_only['text']))
    joe_only = joe_only.withColumn('compound', udf_compound_score(joe_only['scores']))
    joe_only = joe_only.withColumn('comp_score', udf_comp_score(joe_only['compound']))

    # print(trump_only.show(truncate=False))
    # print(joe_only.show(truncate=False))

    joe_pos_only = joe_only[joe_only.comp_score == 'pos']
    joe_neg_only = joe_only[joe_only.comp_score == 'neg']
    joe_neu_only = joe_only[joe_only.comp_score == 'neu']

    trump_pos_only = trump_only[trump_only.comp_score == 'pos']
    trump_neg_only = trump_only[trump_only.comp_score == 'neg']
    trump_neu_only = trump_only[trump_only.comp_score == 'neu']

    # print("Total Trump Tweets \t\t: ", trump_only.count())
    # print("Positive Trump Tweets \t\t: ", trump_pos_only.count())
    # print("Negative Trump Tweets \t\t: ", trump_neg_only.count())
    # print("Neutral Trump Tweets \t\t: ", trump_neu_only.count())

    # print("Total Biden Tweets \t\t: ", joe_only.count())
    # print("Positive Biden Tweets \t\t: ", joe_pos_only.count())
    # print("Negative Biden Tweets \t\t: ", joe_neg_only.count())
    # print("Neutral Biden Tweets \t\t: ", joe_neu_only.count())

    joe_pos_neg_only = joe_only.filter(joe_only['comp_score'] != 'neu')
    trump_pso_neg_only = trump_only.filter(trump_only['comp_score'] != 'neu')
    # print("Total Trump Pos & Neg Tweets Only \t\t: ", trump_pso_neg_only.count())
    # print("Total Biden Pos & Neg Tweets Only  \t\t: ", joe_pos_neg_only.count())

    dt1 = joe_only.groupBy(F.col('location')).agg(F.count('location').alias('joe_total'))
    dt2 = joe_pos_only.groupBy(F.col('location')).agg(F.count('location').alias('joe_pos'))
    dt3 = joe_neg_only.groupBy(F.col('location')).agg(F.count('location').alias('joe_neg'))

    dt4 = trump_only.groupBy(F.col('location')).agg(F.count('location').alias('trump_total'))
    dt5 = trump_pos_only.groupBy(F.col('location')).agg(F.count('location').alias('trump_pos'))
    dt6 = trump_neg_only.groupBy(F.col('location')).agg(F.count('location').alias('trump_neg'))

    # print(dt1.show(truncate=False))
    # print(dt2.show(truncate=False))
    # print(dt3.show(truncate=False))
    # print(dt4.show(truncate=False))
    # print(dt5.show(truncate=False))
    # print(dt6.show(truncate=False))

    # print(dt1.count())
    # print(dt2.count())

    dfs = [dt1, dt2, dt3, dt4, dt5, dt6]
    df_final = reduce(lambda left, right: DataFrame.join(left, right, on='location'), dfs)
    df_final = df_final.sort(F.col('joe_total').asc())
    # print(df_final.show(truncate=False))

    df_per = df_final
    df_per = df_per.withColumn('Joe Pos %', ((df_final['joe_pos'] / df_final['joe_total']) * 100))
    df_per = df_per.withColumn('Joe Neg %', ((df_final['joe_neg'] / df_final['joe_total']) * 100))
    df_per = df_per.withColumn('Trump Pos %', ((df_final['trump_pos'] / df_final['trump_total']) * 100))
    df_per = df_per.withColumn('Trump Neg %', ((df_final['trump_neg'] / df_final['trump_total']) * 100))

    df_per = df_per.withColumn("prediction", when((df_per['Joe Pos %'] > df_per['Trump Pos %']) , "Biden").
                               when((df_per['Joe Pos %'] < df_per['Trump Pos %']) , "Trump").otherwise('Both'))
    print(df_per.show(truncate=False))

    # write to pickle file
    # df_per.rdd.saveAsPickleFile('final_prediction_df.pkl')

    # Read from pickle file
    """for obj in sparkpickle.load_gen("final_prediction_df.pkl"):
        print(obj)"""

if __name__ == '__main__':
    main()
