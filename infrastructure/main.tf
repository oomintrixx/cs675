# infrastructure/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "ds"
}

locals {
  bucket_name      = "ds-${var.student_id}-workspace"
  athena_workgroup = "ds-${var.student_id}"
  glue_db          = "ds_${replace(var.student_id, "-", "_")}"
}

resource "aws_s3_bucket" "workspace" {
  bucket        = local.bucket_name
  force_destroy = true
  tags = { Owner = var.student_id, Project = "cs675" }
}

resource "aws_s3_bucket_versioning" "workspace" {
  bucket = aws_s3_bucket.workspace.id
  versioning_configuration { status = "Disabled" }
}

resource "aws_s3_object" "athena_results_prefix" {
  bucket  = aws_s3_bucket.workspace.id
  key     = "athena-results/.keep"
  content = ""
}

resource "aws_athena_workgroup" "main" {
  name = local.athena_workgroup
  configuration {
    result_configuration {
      output_location = "s3://${aws_s3_bucket.workspace.bucket}/athena-results/"
    }
    bytes_scanned_cutoff_per_query = 10737418240  # 10 GB scan cap
  }
  tags = { Owner = var.student_id }
}

resource "aws_glue_catalog_database" "main" {
  name = local.glue_db
}

resource "aws_emrserverless_application" "spark" {
  name          = "ds-${var.student_id}-spark"
  release_label = "emr-7.1.0"
  type          = "SPARK"

  maximum_capacity {
    cpu    = "8 vCPU"
    memory = "32 GB"
  }

  auto_stop_configuration {
    enabled              = true
    idle_timeout_minutes = 15
  }

  tags = { Owner = var.student_id }
}
