import sys
import boto3
from botocore.exceptions import ClientError
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# ---------------------------------------------------------------------------
# Resolve Glue job parameters — SOURCE_BUCKET and OUTPUT_BUCKET must be
# defined as job parameters in the Glue console / CLI / CloudFormation.
# This replaces the previous hardcoded s3://source-bucket and
# s3://output-bucket string literals.
# ---------------------------------------------------------------------------
args = getResolvedOptions(
    sys.argv,
    ['JOB_NAME', 'SOURCE_BUCKET', 'OUTPUT_BUCKET']
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

source_bucket = args['SOURCE_BUCKET']
output_bucket = args['OUTPUT_BUCKET']

orders_path   = f"s3://{source_bucket}/orders_2024/"
refunds_path  = f"s3://{source_bucket}/refunds_2024/"
output_path   = f"s3://{output_bucket}/combined_transactions/"

# ---------------------------------------------------------------------------
# Pre-flight S3 accessibility check
# Validate that the source bucket/prefixes are reachable before launching
# any Spark work.  A 403 here produces a clear, human-readable error message
# instead of a multi-kilobyte Java stack trace.
# ---------------------------------------------------------------------------
def preflight_s3_check(bucket: str, prefixes: list) -> None:
    """
    Verify S3 bucket accessibility and confirm each prefix has at least one
    object.  Raises a RuntimeError with a descriptive message on any failure
    so the Glue job is marked FAILED immediately with actionable output.
    """
    s3_client = boto3.client("s3")

    # Check bucket-level access
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("403", "AccessDenied"):
            raise RuntimeError(
                f"Pre-flight check FAILED: Access Denied (403) on bucket "
                f"'{bucket}'. Ensure the Glue IAM role has s3:ListBucket and "
                f"s3:GetObject permissions on this bucket."
            ) from e
        elif error_code in ("404", "NoSuchBucket"):
            raise RuntimeError(
                f"Pre-flight check FAILED: Bucket '{bucket}' does not exist."
            ) from e
        else:
            raise RuntimeError(
                f"Pre-flight check FAILED: Unexpected error accessing bucket "
                f"'{bucket}': {e}"
            ) from e

    # Check prefix-level access
    for prefix in prefixes:
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, MaxKeys=1
            )
            if response.get("KeyCount", 0) == 0:
                raise RuntimeError(
                    f"Pre-flight check FAILED: No objects found under "
                    f"s3://{bucket}/{prefix}. Verify the path and that data "
                    f"has been deposited before this job runs."
                )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("403", "AccessDenied"):
                raise RuntimeError(
                    f"Pre-flight check FAILED: Access Denied (403) listing "
                    f"s3://{bucket}/{prefix}. Ensure the Glue IAM role has "
                    f"s3:ListBucket with prefix condition '{prefix}' and "
                    f"s3:GetObject on arn:aws:s3:::{bucket}/{prefix}*."
                ) from e
            else:
                raise RuntimeError(
                    f"Pre-flight check FAILED: Unexpected error listing "
                    f"s3://{bucket}/{prefix}: {e}"
                ) from e


try:
    preflight_s3_check(
        bucket=source_bucket,
        prefixes=["orders_2024/", "refunds_2024/"]
    )
except RuntimeError as preflight_err:
    logger.error(str(preflight_err))
    sys.exit(1)

# ---------------------------------------------------------------------------
# Read source datasets with explicit error handling
# ---------------------------------------------------------------------------
try:
    df_orders = spark.read.parquet(orders_path)
except Exception as e:
    logger.error(
        f"Failed to read orders dataset from '{orders_path}': {e}"
    )
    sys.exit(1)

try:
    df_refunds = spark.read.parquet(refunds_path)
except Exception as e:
    logger.error(
        f"Failed to read refunds dataset from '{refunds_path}': {e}"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Schema validation — assert expected columns are present before union
# A missing or renamed column in the source data would otherwise cause a
# silent schema mismatch at the union step.
# ---------------------------------------------------------------------------
EXPECTED_ORDERS_COLS  = {"order_id", "customer_id", "amount", "status",
                          "order_date", "payment_method"}
EXPECTED_REFUNDS_COLS = {"refund_id", "customer_id", "amount", "status",
                          "refund_date", "payment_mode"}

orders_actual  = set(df_orders.columns)
refunds_actual = set(df_refunds.columns)

missing_orders  = EXPECTED_ORDERS_COLS  - orders_actual
missing_refunds = EXPECTED_REFUNDS_COLS - refunds_actual

if missing_orders:
    logger.error(
        f"Schema validation FAILED for orders dataset: missing columns "
        f"{missing_orders}. Found: {orders_actual}"
    )
    sys.exit(1)

if missing_refunds:
    logger.error(
        f"Schema validation FAILED for refunds dataset: missing columns "
        f"{missing_refunds}. Found: {refunds_actual}"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Transformation — select and union
# ---------------------------------------------------------------------------
orders_subset  = df_orders.select(
    "order_id", "customer_id", "amount", "status",
    "order_date", "payment_method"
)
refunds_subset = df_refunds.select(
    "refund_id", "customer_id", "amount", "status",
    "refund_date", "payment_mode"
)

merged_df = orders_subset.union(refunds_subset)

# ---------------------------------------------------------------------------
# Write output with explicit error handling
# ---------------------------------------------------------------------------
try:
    merged_df.write.mode("overwrite").parquet(output_path)
except Exception as e:
    logger.error(
        f"Failed to write merged dataset to '{output_path}': {e}"
    )
    sys.exit(1)

logger.info(
    f"Job completed successfully. Output written to '{output_path}'."
)
job.commit()
