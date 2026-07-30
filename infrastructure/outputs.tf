# infrastructure/outputs.tf
output "bucket_name"      { value = aws_s3_bucket.workspace.bucket }
output "athena_workgroup" { value = aws_athena_workgroup.main.name }
output "glue_database"    { value = aws_glue_catalog_database.main.name }
output "emr_app_id"       { value = aws_emrserverless_application.spark.id }
