import sys
import boto3
from botocore.exceptions import ClientError
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.utils import AnalysisException

# ---------------------------------------------------------------------------
# Resolve Glue job parameters — all S3 paths are passed as job parameters
# rather than being hard-coded, following AWS Glue best-practice for
# environment promotion (dev / staging / prod) without source edits.
# ---------------------------------------------------------------------------
args = getResolvedOptions(
    sys.argv,
    [
        'JOB_NAME',
        'SOURCE_BUCKET',
        'ORDERS_PREFIX',
        'REFUNDS_PREFIX',
        'OUTPUT_BUCKET',
        'OUTPUT_PREFIX',
    ]
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Build fully-qualified S3 paths from resolved parameters
orders_path = f"s3://{args['SOURCE_BUCKET']}/{args['ORDERS_PREFIX']}/"
refunds_path = f"s3://{args['SOURCE_BUCKET']}/{args['REFUNDS_PREFIX']}/"
output_path = f"s3://{args['OUTPUT_BUCKET']}/{args['OUTPUT_PREFIX']}/"


# ---------------------------------------------------------------------------
# Pre-flight S3 access check
# Verifies bucket accessibility and prefix reachability via boto3 BEFORE
# Spark attempts to read.  Surfaces a clear Python-level RuntimeError in
# CloudWatch Logs instead of a verbose Java AccessDeniedException.
# ---------------------------------------------------------------------------
def check_s3_prefix_accessible(bucket: str, prefix: str) -> None:
    """
    Raises RuntimeError with an actionable message if the Glue execution
    role cannot access s3://<bucket>/<prefix>.

    Checks performed:
      1. s3_client.head_bucket()  — verifies s3:ListBucket permission
      2. s3_client.list_objects_v2(MaxKeys=1) — verifies object listing
    """
    s3_client = boto3.client('s3')

    # 1. Bucket-level access check
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        error_code = exc.response['Error']['Code']
        raise RuntimeError(
            f"Pre-flight check failed: cannot access bucket 's3://{bucket}'. "
            f"AWS error code: {error_code}. "
            f"Ensure the Glue IAM execution role has s3:ListBucket on "
            f"arn:aws:s3:::{bucket}."
        ) from exc

    # 2. Prefix-level listing check
    try:
        s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    except ClientError as exc:
        error_code = exc.response['Error']['Code']
        raise RuntimeError(
            f"Pre-flight check failed: cannot list objects under "
            f"'s3://{bucket}/{prefix}'. "
            f"AWS error code: {error_code}. "
            f"Ensure the Glue IAM execution role has s3:GetObject on "
            f"arn:aws:s3:::{bucket}/{prefix}/*."
        ) from exc


# Run pre-flight checks for both source prefixes before any Spark read
print(f"[PRE-FLIGHT] Checking access to {orders_path}")
check_s3_prefix_accessible(args['SOURCE_BUCKET'], args['ORDERS_PREFIX'])
print(f"[PRE-FLIGHT] Checking access to {refunds_path}")
check_s3_prefix_accessible(args['SOURCE_BUCKET'], args['REFUNDS_PREFIX'])
print("[PRE-FLIGHT] S3 access checks passed.")


# ---------------------------------------------------------------------------
# Read orders
# Wrapped in try/except so that any S3A-layer failure surfaces as a clear
# error in the Glue job run history and CloudWatch Logs.
# ---------------------------------------------------------------------------
try:
    df_orders = spark.read.parquet(orders_path)
    print(f"[READ] Successfully read orders from {orders_path}")
except AnalysisException as exc:
    print(f"[ERROR] Failed to read orders Parquet from {orders_path}: {exc}")
    job.commit()
    sys.exit(1)
except Exception as exc:
    print(f"[ERROR] Unexpected error reading orders from {orders_path}: {exc}")
    job.commit()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Read refunds
# ---------------------------------------------------------------------------
try:
    df_refunds = spark.read.parquet(refunds_path)
    print(f"[READ] Successfully read refunds from {refunds_path}")
except AnalysisException as exc:
    print(f"[ERROR] Failed to read refunds Parquet from {refunds_path}: {exc}")
    job.commit()
    sys.exit(1)
except Exception as exc:
    print(f"[ERROR] Unexpected error reading refunds from {refunds_path}: {exc}")
    job.commit()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------
orders_subset = df_orders.select(
    "order_id",
    "customer_id",
    "amount",
    "status",
    "order_date",
    "payment_method",
)

# Rename 'payment_mode' -> 'payment_method' on the refunds side so that
# the schema aligns before the union.  Without this rename, a positional
# union() silently mis-aligns the payment columns and produces corrupted
# output.  Using unionByName() after the rename makes the intent explicit.
refunds_subset = (
    df_refunds
    .withColumnRenamed("payment_mode", "payment_method")
    .select(
        "refund_id",
        "customer_id",
        "amount",
        "status",
        "refund_date",
        "payment_method",
    )
)


# ---------------------------------------------------------------------------
# Union — use unionByName with allowMissingColumns=True so that any column
# present in one frame but absent in the other is null-filled rather than
# positionally mis-aligned.
# ---------------------------------------------------------------------------
merged_df = orders_subset.unionByName(refunds_subset, allowMissingColumns=True)


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
try:
    merged_df.write.mode("overwrite").parquet(output_path)
    print(f"[WRITE] Successfully wrote combined transactions to {output_path}")
except Exception as exc:
    print(f"[ERROR] Failed to write output to {output_path}: {exc}")
    job.commit()
    sys.exit(1)

job.commit()
