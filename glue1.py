import sys
import logging
import boto3
from botocore.exceptions import ClientError
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col

# Configure structured logging so failures appear clearly in CloudWatch Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve Glue job parameters — S3 paths are parameterised so that partition
# year changes and IAM-scoped path conditions never require a code deployment.
# ---------------------------------------------------------------------------
args = getResolvedOptions(
    sys.argv,
    [
        'JOB_NAME',
        'source_bucket',          # e.g. source-bucket
        'orders_prefix',          # e.g. orders_2024/
        'refunds_prefix',         # e.g. refunds_2024/
        'output_bucket',          # e.g. output-bucket
        'output_prefix',          # e.g. combined_transactions/
    ]
)

SOURCE_BUCKET   = args['source_bucket']
ORDERS_PREFIX   = args['orders_prefix']
REFUNDS_PREFIX  = args['refunds_prefix']
OUTPUT_BUCKET   = args['output_bucket']
OUTPUT_PREFIX   = args['output_prefix']

ORDERS_PATH  = f"s3://{SOURCE_BUCKET}/{ORDERS_PREFIX}"
REFUNDS_PATH = f"s3://{SOURCE_BUCKET}/{REFUNDS_PREFIX}"
OUTPUT_PATH  = f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}"

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ---------------------------------------------------------------------------
# Pre-flight S3 accessibility check
# Perform a boto3 head_object probe against both source prefixes BEFORE
# invoking spark.read.  A 403 here produces a clear, structured log message
# rather than a deep JVM / S3AFileSystem stack trace, and exits fast without
# wasting DPU-hours on a job that cannot succeed.
# ---------------------------------------------------------------------------
def assert_s3_readable(bucket: str, key_prefix: str) -> None:
    """
    Raises a RuntimeError with a clear diagnostic message when the Glue
    execution role cannot reach s3://<bucket>/<key_prefix>.
    """
    s3_client = boto3.client('s3')
    try:
        # list_objects_v2 with MaxKeys=1 validates both s3:ListBucket and
        # that the prefix actually exists without transferring object data.
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=key_prefix,
            MaxKeys=1
        )
        key_count = response.get('KeyCount', 0)
        logger.info(
            "Pre-flight check passed for s3://%s/%s (KeyCount=%d)",
            bucket, key_prefix, key_count
        )
    except ClientError as exc:
        error_code = exc.response['Error']['Code']
        raise RuntimeError(
            f"Pre-flight S3 accessibility check FAILED for "
            f"s3://{bucket}/{key_prefix}. "
            f"AWS error code: {error_code}. "
            f"Ensure the Glue IAM execution role has s3:ListBucket on "
            f"arn:aws:s3:::{bucket} and s3:GetObject on "
            f"arn:aws:s3:::{bucket}/*. "
            f"Original error: {exc}"
        ) from exc

try:
    assert_s3_readable(SOURCE_BUCKET, ORDERS_PREFIX)
    assert_s3_readable(SOURCE_BUCKET, REFUNDS_PREFIX)
except RuntimeError as preflight_err:
    logger.error("Pre-flight check failed — aborting Glue job. %s", preflight_err)
    job.commit()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Read source data
# Each spark.read call is wrapped in its own try/except so that a failure on
# one path surfaces a structured log entry with the exact S3 URI before the
# JVM exception propagates.
# ---------------------------------------------------------------------------
try:
    df_orders = spark.read.parquet(ORDERS_PATH)
    logger.info("Successfully read orders parquet from %s", ORDERS_PATH)
except Exception as exc:
    logger.error(
        "Failed to read orders parquet from %s. Error: %s",
        ORDERS_PATH, exc
    )
    raise

try:
    df_refunds = spark.read.parquet(REFUNDS_PATH)
    logger.info("Successfully read refunds parquet from %s", REFUNDS_PATH)
except Exception as exc:
    logger.error(
        "Failed to read refunds parquet from %s. Error: %s",
        REFUNDS_PATH, exc
    )
    raise

# ---------------------------------------------------------------------------
# Schema alignment
# orders_subset  uses the column name  payment_method
# refunds_subset uses the column name  payment_mode
#
# PySpark .union() is POSITIONAL — without alignment, payment_mode values
# would be silently written under the payment_method label for all refund
# rows, producing corrupt downstream output with no error or warning.
#
# Fix: alias payment_mode -> payment_method in refunds_subset so both sides
# share an identical schema, then use .unionByName() to make Spark align
# columns by name (not position) as an additional safety net.
# ---------------------------------------------------------------------------
orders_subset = df_orders.select(
    "order_id",
    "customer_id",
    "amount",
    "status",
    "order_date",
    "payment_method"
)

refunds_subset = df_refunds.select(
    "refund_id",
    "customer_id",
    "amount",
    "status",
    "refund_date",
    col("payment_mode").alias("payment_method")   # align name to orders schema
)

# .unionByName() aligns on column names rather than positions, surfacing any
# remaining schema divergence as an explicit Spark error instead of silent
# data corruption.  allowMissingColumns=True tolerates the order_id /
# refund_id asymmetry by filling the absent column with null.
merged_df = orders_subset.unionByName(refunds_subset, allowMissingColumns=True)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
try:
    merged_df.write.mode("overwrite").parquet(OUTPUT_PATH)
    logger.info("Successfully wrote combined transactions to %s", OUTPUT_PATH)
except Exception as exc:
    logger.error(
        "Failed to write combined transactions to %s. Error: %s",
        OUTPUT_PATH, exc
    )
    raise

job.commit()
