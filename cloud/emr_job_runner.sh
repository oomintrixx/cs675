#!/bin/bash
# cloud/emr_job_runner.sh
set -e

BUCKET="${CS675_BUCKET:?}"
PROFILE="${AWS_PROFILE:-ds}"
APP_ID=$(cd infrastructure && terraform output -raw emr_app_id)
ROLE_ARN="${EMR_ROLE_ARN:-$(cd infrastructure && terraform output -raw emr_role_arn)}"
SCRIPT="${1:?Usage: $0 <script_name>  e.g. 02_preprocess_cloud}"

aws s3 cp "cloud/${SCRIPT}.py"        "s3://${BUCKET}/scripts/${SCRIPT}.py"        --profile "${PROFILE}"
aws s3 cp "work/preprocess_steps.py"  "s3://${BUCKET}/scripts/preprocess_steps.py" --profile "${PROFILE}"
aws s3 cp "work/analytics.py"         "s3://${BUCKET}/scripts/analytics.py"        --profile "${PROFILE}"
aws s3 cp "work/ml_features.py"       "s3://${BUCKET}/scripts/ml_features.py"      --profile "${PROFILE}"

echo "Submitting ${SCRIPT} to EMR Serverless..."
RUN_ID=$(aws emr-serverless start-job-run \
  --application-id "${APP_ID}" \
  --execution-role-arn "${ROLE_ARN}" \
  --name "cs675-${SCRIPT}" \
  --job-driver "{
    \"sparkSubmit\": {
      \"entryPoint\": \"s3://${BUCKET}/scripts/${SCRIPT}.py\",
      \"sparkSubmitParameters\": \"--py-files s3://${BUCKET}/scripts/preprocess_steps.py,s3://${BUCKET}/scripts/analytics.py,s3://${BUCKET}/scripts/ml_features.py --conf spark.emr-serverless.driverEnv.CS675_BUCKET=${BUCKET} --conf spark.executorEnv.CS675_BUCKET=${BUCKET} --conf spark.driver.cores=1 --conf spark.driver.memory=2g --conf spark.executor.cores=1 --conf spark.executor.memory=2g --conf spark.executor.instances=2 --conf spark.dynamicAllocation.enabled=false\"
    }
  }" \
  --configuration-overrides "{
    \"monitoringConfiguration\": {
      \"s3MonitoringConfiguration\": { \"logUri\": \"s3://${BUCKET}/emr-logs/\" }
    }
  }" \
  --profile "${PROFILE}" \
  --query "jobRunId" --output text)

echo "Job ID: ${RUN_ID}"
while true; do
  STATUS=$(aws emr-serverless get-job-run \
    --application-id "${APP_ID}" --job-run-id "${RUN_ID}" \
    --profile "${PROFILE}" --query "jobRun.state" --output text)
  echo "  Status: ${STATUS}"
  [[ "$STATUS" == "SUCCESS" || "$STATUS" == "FAILED" || "$STATUS" == "CANCELLED" ]] && break
  sleep 30
done

[[ "$STATUS" != "SUCCESS" ]] && { echo "FAILED. Logs: s3://${BUCKET}/emr-logs/"; exit 1; }
echo "Done: ${SCRIPT}"
