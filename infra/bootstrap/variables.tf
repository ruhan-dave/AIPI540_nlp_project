variable "region" {
  description = "The AWS region the Terraform state bucket lives in."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for the state bucket name. Keep it short — S3 bucket names are limited to 63 characters."
  type        = string
  default     = "reviewassist"
}

variable "state_bucket_name" {
  description = <<-EOT
    An explicit name for the state bucket. Leave empty to derive one from
    `name_prefix` and a random suffix, which guarantees the name is globally
    unique.
  EOT
  type        = string
  default     = ""
}

variable "allowed_principals" {
  description = <<-EOT
    IAM principal ARNs allowed to read and write the Terraform state.

    An empty list (the default) allows every principal in the account that IAM
    authorizes, which keeps the bootstrap usable. Set it to the deployment
    role's ARN — for example
    `arn:aws:iam::123456789012:role/TerraformDeploy` — to restrict state access
    to that role.
  EOT
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags applied to the state bucket."
  type        = map(string)
  default = {
    Project = "reviewassist"
    Managed = "terraform"
  }
}
