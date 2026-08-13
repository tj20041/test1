import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df_orders = spark.read.parquet("s3://source-bucket/orders_2024/")
df_refunds = spark.read.parquet("s3://source-bucket/refunds_2024/")

orders_subset = df_orders.select("order_id", "customer_id", "amount", "status", "order_date", "payment_method")
refunds_subset = df_refunds.select("refund_id", "customer_id", "amount", "status", "refund_date", "payment_mode")

merged_df = orders_subset.union(refunds_subset)

merged_df.write.mode("overwrite").parquet("s3://output-bucket/combined_transactions/")
job.commit()
