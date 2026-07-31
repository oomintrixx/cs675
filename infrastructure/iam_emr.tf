# infrastructure/iam_emr.tf
data "aws_iam_policy_document" "emr_serverless_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["emr-serverless.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = [aws_emrserverless_application.spark.arn]
    }
  }
}

data "aws_caller_identity" "current" {}

resource "aws_iam_role" "emr_serverless_execution" {
  name               = "EMRServerlessExecutionRole"
  assume_role_policy = data.aws_iam_policy_document.emr_serverless_trust.json
  tags               = { Owner = var.student_id }
}

data "aws_iam_policy_document" "emr_serverless_s3" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${aws_s3_bucket.workspace.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.workspace.arn]
  }
}

resource "aws_iam_role_policy" "emr_serverless_s3" {
  name   = "emr-serverless-workspace-s3"
  role   = aws_iam_role.emr_serverless_execution.id
  policy = data.aws_iam_policy_document.emr_serverless_s3.json
}
