from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql import DataFrame
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, Row, ArrayType, StringType
import pyspark.sql.functions as F
from pyspark.sql.functions import *
import json

spark = SparkSession.builder.appName("Twitter Analysis").getOrCreate()
sc = spark.sparkContext

tweets=spark.read.parquet("/FileStore/tables/est_tweets/")
tweets=tweets.drop('time','location').withColumnRenamed('new_location','state').withColumn("date_only", F.to_date(F.col("est_time")))

#tweets = tweets[(tweets['est_time'] > '2020-09-10 07:00:00') & (tweets['est_time'] < '2020-11-02 11:59:00')]
tweets.printSchema()
tweets.count()

joe_only = tweets.filter((tweets['text'].rlike("[Jj]oe|[Bb]iden") == True) & (tweets['text'].rlike("[Dd]onald|[Tt]rump") == False))
# print("Only Biden Tweets \t\t: ", joe_only.count())
trump_only = tweets.filter((tweets['text'].rlike("[Jj]oe|[Bb]iden") == False) & (tweets['text'].rlike("[Dd]onald|[Tt]rump") == True))
# print("Only Donald Trump Tweets \t\t: ", trump_only.count())
joe_and_trump = tweets.filter((tweets['text'].rlike("[Dd]onald|[Tt]rump")) & (tweets['text'].rlike("[Jj]oe|[Bb]iden")))
# print("Both Joe_Biden & Trump Tweets \t\t: ", joe_and_trump.count())
not_joe_trump = tweets.filter(~(tweets['text'].rlike("[Dd]onald|[Tt]rump")) & ~(tweets['text'].rlike("[Jj]oe|[Bb]iden")))
# print("Tweets without Joe_Biden & Trump \t: ", not_joe_trump.count())
print("Total Tweets \t\t\t\t: ",tweets.count())
print("Joe Biden Tweets \t\t\t: ",joe_only.count())
print("Donald Trump Tweets \t\t\t: ",trump_only.count())
print("Tweets contaning both candidate names \t\t: ", joe_and_trump.count())
print("Tweets with neither Joe Biden nor Donald Trump names \t: ",not_joe_trump.count())

#joeBiden tweets excluding trump
Harris_only = tweets.filter((tweets['text'].rlike("[Kk]amala|[Hh]arris") == True) & (tweets['text'].rlike("[Mm]ike|[Pp]ence") == False))
Pence_only = tweets.filter((tweets['text'].rlike("[Mm]ike|[Pp]ence") == True) & (tweets['text'].rlike("[Kk]amala|[Hh]arris") == False))
Harris_and_Pence = tweets.filter((tweets['text'].rlike("[Mm]ike|[Pp]ence")) & (tweets['text'].rlike("[Kk]amala|[Hh]arris")))
not_Harris_Pence = tweets.filter(~(tweets['text'].rlike("[Mm]ike|[Pp]ence")) & ~(tweets['text'].rlike("[Kk]amala|[Hh]arris")))
#Kamala Harris Mike Pence
print("Total Tweets \t\t\t\t: ",len(tweets))
print("Only Kamala Harris Tweets \t\t\t: ",Harris_only.count())
print("Only Mike Pence Tweets \t\t: ",Pence_only.count())
print("Tweets contaning both VP candidate names \t\t: ",Harris_and_Pence.count())
print("Tweets without Kamala Harris nor Mike Pence names \t: ",not_Harris_Pence.count())

sid = SentimentIntensityAnalyzer()
udf_priority_score = udf(lambda x: sid.polarity_scores(x), returnType=StringType())  # Define UDF function
udf_compound_score = udf(lambda score_dict: score_dict['compound'])
udf_comp_score = udf(lambda c: 'pos' if c >= 0.05 else ('neu' if (c > -0.05 and c < 0.05) else 'neg'))

trump_only = trump_only.withColumn('scores', udf_priority_score(trump_only['text']))
trump_only = trump_only.withColumn('compound', udf_compound_score(trump_only['scores']))
trump_only = trump_only.withColumn('comp_score', udf_comp_score(trump_only['compound']))
joe_only = joe_only.withColumn('scores', udf_priority_score(joe_only['text']))
joe_only = joe_only.withColumn('compound', udf_compound_score(joe_only['scores']))
joe_only = joe_only.withColumn('comp_score', udf_comp_score(joe_only['compound']))

trump_only.printSchema()

joe_pos_only = joe_only[joe_only.comp_score == 'pos']
joe_neg_only = joe_only[joe_only.comp_score == 'neg']
joe_neu_only = joe_only[joe_only.comp_score == 'neu']
trump_pos_only = trump_only[trump_only.comp_score == 'pos']
trump_neg_only = trump_only[trump_only.comp_score == 'neg']
trump_neu_only = trump_only[trump_only.comp_score == 'neu']
trump_pos_only.printSchema()

joe_pos_neg_only = joe_only.filter(joe_only['comp_score'] != 'neu')
trump_pos_neg_only = trump_only.filter(trump_only['comp_score'] != 'neu')


print(trump_pos_neg_only.count())

dt1 = joe_pos_neg_only.groupBy(F.col('state')).agg(F.count('state').alias('joe_total'))
dt2 = joe_pos_only.groupBy(F.col('state')).agg(F.count('state').alias('joe_pos'))
dt3 = joe_neg_only.groupBy(F.col('state')).agg(F.count('state').alias('joe_neg'))

dt4 = trump_pos_neg_only.groupBy(F.col('state')).agg(F.count('state').alias('trump_total'))
dt5 = trump_pos_only.groupBy(F.col('state')).agg(F.count('state').alias('trump_pos'))
dt6 = trump_neg_only.groupBy(F.col('state')).agg(F.count('state').alias('trump_neg'))

dfs = [dt1, dt2, dt3, dt4, dt5, dt6]
#df_final = reduce(lambda left, right: left.join(right,["date_only","state"]), dfs)
df_final = reduce(lambda left, right: DataFrame.join(left, right, on='state'), dfs)
df_final = df_final.sort(F.col('joe_total').asc())
df_final.write.mode('overwrite').parquet("/FileStore/tables/pre_analysis/")

print(df_final.show())

df=spark.read.parquet("/FileStore/tables/pre_analysis/")
df.show(2000)

df_per=df.withColumn("Joe Pos %", F.round((F.col("joe_pos") / F.col("joe_total"))*100,2))
df_per=df_per.withColumn("Joe Neg %", F.round((F.col("joe_neg") / F.col("joe_total"))*100,2))
df_per=df_per.withColumn("Trump Pos %", F.round((F.col("trump_pos") / F.col("trump_total"))*100,2))
df_per=df_per.withColumn("Trump Neg %", F.round((F.col("trump_neg") / F.col("trump_total"))*100,2))
df_per=df_per.select(col("state").alias("State"),"Joe Pos %","Joe Neg %","Trump Pos %","Trump Neg %")
df_per.show()

# df_per=df_final.withColumn("Joe Pos %", F.round((F.col("joe_pos") / F.col("joe_total"))*100,2))
df_per = df_per.withColumn("Trump diff",F.round((F.col("Trump Pos %") - F.col("Trump Neg %")),2))
df_per = df_per.withColumn("Biden diff",F.round((F.col("Joe Pos %") - F.col("Joe Neg %")),2))
# df_per = df_per.withColumn('Biden diff',((df_per['Joe Pos %'] - df_per['Joe Neg %'])))
df_per = df_per.withColumn("Who wins", when((df_per['Joe Pos %'] > df_per['Trump Pos %']) , "Biden").
                           when((df_per['Joe Pos %'] < df_per['Trump Pos %']) , "Trump").otherwise('Both'))
#df_per= df_per.sort(F.col('date_only').asc())
print(df_per.show(60,truncate=False))


# write to pickle file
#df_per.rdd.saveAsPickleFile('pre_analysis.pkl')
#df_per=spark.read.parquet("/FileStore/tables/pre_analysis/")
df_per.repartition(1).write.csv(path="/FileStore/pre_analysis.csv", mode="overwrite", header="true")