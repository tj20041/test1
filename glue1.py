import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import lit

# Resolve Glue job parameters — S3 paths are externalised so that IAM policies
# can be scoped per environment and paths can be changed without code edits.
args = getResolvedOptions(
    sys.argv,
    ['JOB_NAME', 'SOURCE_ORDERS_PATH', 'SOURCE_REFUNDS_PATH', 'OUTPUT_PATH']
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ---------------------------------------------------------------------------
# S3 path pre-validation
# Perform a lightweight boto3 accessibility check before issuing Spark reads.
# This produces a clear, human-readable CloudWatch message on permission or
# path problems instead of a multi-screen Java stack trace.
# NOTE: The Glue execution role must have s3:GetObject, s3:GetObjectVersion,
#       s3:ListBucket, and s3:GetBucketLocation granted on both source-bucket
#       and output-bucket (and their /* prefixes) for this job to succeed.
#       Attach the required IAM policy to the role shown in the Glue console
#       under Job details > IAM role before re-running this job.
# ---------------------------------------------------------------------------

def _parse_s3_path(s3_path):
    """Return (bucket, prefix) from an s3://bucket/prefix string."""
    path = s3_path.replace("s3://", "", 1)
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _validate_s3_path(s3_path, label):
    """
    Use boto3 to verify the Glue execution role can list the given S3 prefix.
    Raises SystemExit(1) with a descriptive message if access is denied or the
    prefix does not exist, avoiding the opaque Java AccessDeniedException.
    """
    s3_client = boto3.client("s3")
    bucket, prefix = _parse_s3_path(s3_path)
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1
        )
        key_count = response.get("KeyCount", 0)
        if key_count == 0:
            print(
                f"[WARN] S3 path '{s3_path}' ({label}) is accessible but "
                f"contains no objects. The Spark read will likely fail or "
                f"return an empty DataFrame."
            )
        else:
            print(f"[INFO] S3 path '{s3_path}' ({label}) is accessible "
                  f"and contains objects. Proceeding with read.")
    except s3_client.exceptions.NoSuchBucket:
        print(
            f"[FATAL] S3 bucket '{bucket}' does not exist "
            f"(path: {s3_path}, label: {label}). "
            f"Verify the bucket name and that it exists in this AWS region."
        )
        raise SystemExit(1)
    except Exception as e:
        error_str = str(e)
        if "AccessDenied" in error_str or "403" in error_str:
            print(
                f"[FATAL] Access Denied (HTTP 403) reading S3 path "
                f"'{s3_path}' ({label}). "
                f"The Glue execution role is missing s3:ListBucket / "
                f"s3:GetObject permissions on 's3://{bucket}'. "
                f"Attach the required IAM policy to the Glue role before "
                f"re-running this job. Original error: {e}"
            )
        else:
            print(
                f"[FATAL] Cannot access S3 path '{s3_path}' ({label}). "
                f"Error: {e}"
            )
        raise SystemExit(1)


_validate_s3_path(args['SOURCE_ORDERS_PATH'], "SOURCE_ORDERS_PATH")
_validate_s3_path(args['SOURCE_REFUNDS_PATH'], "SOURCE_REFUNDS_PATH")

# ---------------------------------------------------------------------------
# Read source datasets
# ---------------------------------------------------------------------------

try:
    df_orders = spark.read.parquet(args['SOURCE_ORDERS_PATH'])
except Exception as e:
    print(
        f"[FATAL] Cannot read orders parquet from "
        f"'{args['SOURCE_ORDERS_PATH']}'. Error: {e}"
    )
    raise SystemExit(1)

try:
    df_refunds = spark.read.parquet(args['SOURCE_REFUNDS_PATH'])
except Exception as e:
    print(
        f"[FATAL] Cannot read refunds parquet from "
        f"'{args['SOURCE_REFUNDS_PATH']}'. Error: {e}"
    )
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Select and align columns before union
#
# orders columns : order_id, customer_id, amount, status, order_date,
#                  payment_method
# refunds columns: refund_id, customer_id, amount, status, refund_date,
#                  payment_mode
#
# PySpark DataFrame.union() merges by position, NOT by column name.  A naive
# union therefore silently maps refund_date -> order_date and
# payment_mode -> payment_method, corrupting the output data.
#
# Fix:
#   1. Rename differing refunds columns to match the orders schema.
#   2. Add a 'source_type' discriminator column to both subsets so downstream
#      consumers can distinguish orders from refunds in the combined table.
#   3. Use unionByName() to enforce name-based merging rather than positional.
# ---------------------------------------------------------------------------

orders_subset = (
    df_orders
    .select("order_id", "customer_id", "amount", "status",
            "order_date", "payment_method")
    .withColumn("source_type", lit("order"))
)

refunds_subset = (
    df_refunds
    .select("refund_id", "customer_id", "amount", "status",
            "refund_date", "payment_mode")
    # Rename to match the orders column names so unionByName works correctly
    # and no data is silently misaligned.
    .withColumnRenamed("refund_id", "order_id")
    .withColumnRenamed("refund_date", "order_date")
    .withColumnRenamed("payment_mode", "payment_method")
    .withColumn("source_type", lit("refund"))
)

# unionByName raises an AnalysisException if schemas still differ, providing
# an explicit error rather than silent data corruption.
merged_df = orders_subset.unionByName(refunds_subset)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

try:
    merged_df.write.mode("overwrite").parquet(args['OUTPUT_PATH'])
except Exception as e:
    print(
        f"[FATAL] Cannot write output parquet to "
        f"'{args['OUTPUT_PATH']}'. Error: {e}"
    )
    raise SystemExit(1)

print(
    f"[INFO] Job complete. Combined transactions written to "
    f"'{args['OUTPUT_PATH']}'"
)

job.commit()
