# infrastructure/variables.tf
variable "student_id" {
  description = "Your student ID — used to name all AWS resources"
  type        = string
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}
