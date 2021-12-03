# ================== Toturial:
"""
https://towardsdatascience.com/a-neanderthals-guide-to-apache-spark-in-python-9ef1f156d427
"""


# ================== install:
# Install java 
"""
install JDK version should be 8 or 11
"""

# For Anaconda:
"""
!conda install -c cyclus java-jdk
!pip install pyspark
!pip install pyspark --upgrade

"""

# Setup Variables
"""
import os
os.environ["JAVA_HOME"] = @todo: "C:\Program Files\Java\jdk-11.0.12" 
os.environ['PYSPARK_PYTHON'] = 'python'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'jupyter'
os.environ['PYSPARK_DRIVER_PYTHON_OPTS'] = 'notebook' 

"""


# ================== Libraries: 
# for pyspark 
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext, SparkConf

# related 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()



# ==================  Initilize:
# Without Seting memory and CPU cores
conf  = SparkConf().set("spark.ui.port", "4050")
sc    = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
spark

# Sets memory limit on driver and to use all CPU cores
conf = SparkConf()\
        .set('spark.ui.port', '4050') \
        .set('spark.driver.memory', '4g') \
        .setMaster('local[*]')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()



# ==================  Initilize Dynamic:
sc = None
def configured_spark_session(parallelism=None, shuffle_partitions=None, mem='4g'):
    # Docs: Once a SparkConf object is passed to Spark, it is 
    # cloned and can no longer be modified by the user. 
    # Spark does not support modifying the configuration at runtime.
    # Additionally, spark.driver.memory is honored only the FIRST time
    # a sparkcontext is created (this is when the JVM starts). You have 
    # to restart the kernel (killing the JVK) for mem argument to take 
    # effect, even though the spark config will show a different value, 
    # it's not actually the case (see Executors tab).
    # We can change parallelism and shuffle_partitions with this function, but
    # note this restarts (and thus clears) current spark session
    global sc
    conf = SparkConf() \
            .set('spark.ui.port', '4051') \
            .set('spark.driver.memory', mem) \
            .set('spark.sql.adaptive.enabled', False) \
            .setMaster('local[*]') # Uses all cores, can be set to integer
    if parallelism: conf = conf.set('spark.default.parallelism', parallelism)
    if shuffle_partitions: conf = conf.set('spark.sql.shuffle.partitions', shuffle_partitions)
    if sc: sc.stop()
    sc = pyspark.SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()
    return spark

spark = configured_spark_session()
sc.getConf().getAll()
spark



# ==================  Methods:
# spark.read: 
df = spark.read \
    .option('header', True) \
    .option('inferSchema', True) \
    .csv('titanic_full.csv')

rc = spark.read\
    .csv('titanic_full.csv', header=True)

df_aircraft = spark.read.json('Aircraft_Glossary.json')

# .filter:
df = df.filter(F.col('Age').isNotNull())

rc.filter(F.col('DISTRICT') == '22') \
    .groupBy(F.col('PRIMARY_TYPE')) \
    .count() \
    .sort(F.desc('count')) \
    .show(3)

cond = (F.col('MissionDate') == '1966-06-29') & (F.col('TargetCountry') == 'NORTH VIETNAM')
df.filter(cond)

# F.udf:
""" define the function """
def age_bracket(age):
    for bracket in range(10, 150, 10):
        if age < bracket:
            return f"{bracket-10}-{bracket-1}"

""" add it to udf """
age_bracket_udf = F.udf(age_bracket)

""" use it """
df = df.withColumn('AgeBracketUDF', age_bracket_udf(F.col('Age'))) \
       .withColumn('AgeBracketDiv', (F.col('Age') / 10).cast('integer')*10)

# df.select
df.select('Age', 'AgeBracketUDF', 'AgeBracketDiv') \
    .show() 

# .groupBy 
df.groupBy('AgeBracketUDF', 'AgeBracketDiv') \
    .agg(F.count('AgeBracketUDF'), F.count('AgeBracketDiv')) \
    .sort('AgeBracketUDF', 'AgeBracketDiv') \
    .collect()

# .explain F.count
df.groupBy('AgeBracketUDF', 'AgeBracketDiv') \
    .agg(F.count('AgeBracketUDF'), F.count('AgeBracketDiv')) \
    .sort('AgeBracketUDF', 'AgeBracketDiv') \
    .explain()

# .toPandas
rc.toPandas()

# .count
rc.count()

# .sort
rc.groupBy('LOCATION_DESCRIPTION') \
    .count() \
    .sort(F.desc('count')) \
    .limit(3) \
    .toPandas()


# .set_index F.reverse
rc.select(F.reverse(F.split('BLOCK', ' '))[0].alias('suffix')) \
    .groupby('suffix') \
    .count() \
    .toPandas() \
    .set_index('suffix') \
    .T

# F.lower F.sum F.col .select .agg .endswith .alias .cast 
rc.select(F.lower('BLOCK').alias('block')) \
    .select(
        F.col('block').endswith('ave').alias('AVE').cast('integer'),
        F.col('block').endswith('av').alias('AV').cast('integer'),
        F.col('block').endswith('blvd').alias('BLVD').cast('integer'),
        F.col('block').endswith('st').alias('ST').cast('integer'),
        F.col('block').endswith('dr').alias('DR').cast('integer'),
        F.col('block').endswith('pl').alias('PL').cast('integer'),
        F.col('block').endswith('rd').alias('RD').cast('integer'),
        F.col('block').endswith('pkwy').alias('PKWY').cast('integer'),
        F.col('block').endswith('ct').alias('CT').cast('integer')
        ) \
    .agg(
        (F.sum('AVE') + F.sum('AV')).alias('sum(AVE)'), 
        F.sum('BLVD'), F.sum('ST'), 
        F.sum('DR'), F.sum('PL'), 
        F.sum('RD'), F.sum('PKWY'), 
        F.sum('CT')
        ) \
    .toPandas()


# .printSchema
df_aircraft.printSchema()

# .sort F.asc .plot .barh
_ = df_operations \
        .groupBy('ContryFlyingMission') \
        .count() \
        .sort(F.asc('count')) \
        .toPandas()\
        .set_index('ContryFlyingMission')\
        .plot \
        .barh(figsize=(16, 4), log=True)


# F.to_date
_ = df_operations \
    .select(
        F.col('ContryFlyingMission'), 
        F.to_date(F.col('MissionDate'), 'yyyy-MM-dd').alias('MissionDate')
        )

# F.trunc
df_operations_q2_counts_months_pd = \
    df_operations_q2 \
        .withColumn('MissionYearMonth', F.trunc('MissionDate', 'mm')) \
        .groupBy(['ContryFlyingMission', 'MissionYearMonth']) \
        .count() \
        .sort(F.asc('MissionYearMonth')) \
        .toPandas()

# loop through group by
fig, ax = plt.subplots(figsize=(16,4))
for label, df in df_operations_q2_counts_pd.groupby(by='ContryFlyingMission'):
    ax.plot(df['MissionDate'], df['count'], label=label)
_ = ax.legend()


# .distinct
df_operations_jun_29 \
    .select(F.col('ContryFlyingMission')) \
    .distinct() \
    .show(truncate=False)

# .cach 
df_operations_jun_29 \
    .groupby('ContryFlyingMission') \
    .count() \
    .collect()

df_operations_jun_29 \
    .cache()

df_operations_jun_29 \
    .count()

df_operations_jun_29.groupby('ContryFlyingMission')\
    .count() \
    .collect()

# .write
df_operations_jun_29 \
    .write \
    .json('operations_jun_29.json')

spark.read.json('operations_jun_29.json')

# F.percentile_approx F.mean(
df_operations_jun_29 \
    .groupby('TakeoffLocation') \
    .agg(
        F.mean('TimeOnTarget').alias('avg_tot'), 
        F.percentile_approx(
            'TimeOnTarget',
            [0.25, 0.5, 0.75],
            1000000
            ).alias('quantiles')
        ) \
    .sort(F.asc('avg_tot')) \
    .toPandas()

# .join
df_operations.join(df_aircraft, on='AirCraft', how='inner')


# .join .hint('shuffle_merge')
df_operations \
    .join(df_aircraft.hint('shuffle_merge'), on='AirCraft', how='inner') \
    .groupby('AirCraftType') \
    .agg(F.avg('TimeOnTarget').alias('avg_tot')) \
    .sort(F.asc('avg_tot')) \
    .toPandas()

