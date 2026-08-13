import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# ==========================================
# 0. GLUE INITIALIZATION & LOGGING SETUP
# ==========================================
# Fetch job name passed by AWS Glue framework
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# CloudWatch logs automatically capture output from standard Python logger
logger = logging.getLogger("CustomerDemographicsETL")
logger.setLevel(logging.INFO)

# Attach StreamHandler for explicit CloudWatch/stdout writing
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("Initializing Customer Demographics Glue Job...")

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")
    schema1 = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
    ])

    schema2 = StructType([
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("id", IntegerType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting data into Bronze layer...")
    df1_bronze = spark.createDataFrame([
        (1, "Alice", 25),
        (2, "TestUser", -15),
        (3, "Charlie", 30)
    ], schema=schema1)

    df2_bronze = spark.createDataFrame([
        ("Dave", 22, 4),
        ("BotAccount", -5, 5),
        ("Eve", 28, 6)
    ], schema=schema2)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")
    
    # Safe schema mapping using unionByName instead of positional union
    df_gold = df1_silver.unionByName(df2_silver)
    
    logger.info("Pipeline executed successfully. Output dataframe schema:")
    df_gold.printSchema()
    
    # Materialize execution in logs (replaces Databricks display())
    df_gold.show(truncate=False)

    # Commit Glue Job State
    job.commit()

except Exception as e:
    logger.error("Glue Pipeline failed during execution. Error details: %s", str(e))
    raise
