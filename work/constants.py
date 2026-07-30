# work/constants.py
import os

IS_CLOUD = os.environ.get("CS675_ENV") == "cloud"

LOCAL_TAXI_PATH   = "/home/jovyan/data/taxi/"
LOCAL_OUTPUT_PATH = "/home/jovyan/data/output/"

BUCKET            = os.environ.get("CS675_BUCKET", "ds-student-workspace")
CLOUD_TAXI_PATH   = f"s3://{BUCKET}/data/taxi/"
CLOUD_OUTPUT_PATH = f"s3://{BUCKET}/output/"

TAXI_PATH   = CLOUD_TAXI_PATH   if IS_CLOUD else LOCAL_TAXI_PATH
OUTPUT_PATH = CLOUD_OUTPUT_PATH if IS_CLOUD else LOCAL_OUTPUT_PATH
