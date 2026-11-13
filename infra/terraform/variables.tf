variable "region" {
  description = "The AWS region everything is created in."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for the resource names. Keep it short — ALB and target group names are limited to 32 characters."
  type        = string
  default     = "reviewassist"
}

variable "vpc_id" {
  description = "An existing VPC with at least two public subnets in different Availability Zones. This stack does not create networking."
  type        = string
}

variable "public_subnet_ids" {
  description = "The IDs of at least two existing public subnets that route to an Internet Gateway. The tasks get public IPs so they can reach ECR, CloudWatch and the LLM provider without a NAT gateway."
  type        = list(string)

  validation {
    condition     = length(var.public_subnet_ids) >= 2
    error_message = "At least two public subnets in different Availability Zones are required for the load balancer."
  }
}

variable "api_image_tag" {
  description = "The tag of the backend image in the ECR repository. The image must be pushed before the services are started."
  type        = string
  default     = "v1"
}

variable "api_domain_name" {
  description = "The public DNS name of the API, for example api.reviewassist.example. Point it at the load balancer's DNS name and serve it with the ACM certificate."
  type        = string
}

variable "acm_certificate_arn" {
  description = "The ARN of an ACM certificate in this region covering `api_domain_name`."
  type        = string
}

variable "api_secret_arn" {
  description = <<-EOT
    An existing Secrets Manager secret holding the API bearer token. Leave it
    empty to use the secret this stack creates — Terraform manages the secret's
    metadata and ARN, and the value is seeded out of band.
  EOT
  type        = string
  default     = null
}

variable "provider_secret_arn" {
  description = <<-EOT
    An existing Secrets Manager secret holding the LLM provider key. Leave it
    empty to use the secret this stack creates.
  EOT
  type        = string
  default     = null
}

variable "llm_provider" {
  description = "The provider that splits reviews: `openrouter` or `gemini`. The matching key is read from Secrets Manager."
  type        = string
  default     = "openrouter"

  validation {
    condition     = contains(["openrouter", "gemini"], var.llm_provider)
    error_message = "llm_provider must be either openrouter or gemini."
  }
}

variable "desired_count" {
  description = "How many tasks each service runs. Start at 0 for the first apply: the image does not exist in ECR until it has been pushed."
  type        = number
  default     = 0

  validation {
    condition     = var.desired_count >= 0 && var.desired_count <= 10
    error_message = "desired_count must be between 0 and 10."
  }
}

variable "max_upload_bytes" {
  description = "The largest CSV a single job may carry (50 MiB)."
  type        = number
  default     = 52428800
}

variable "max_reviews_per_job" {
  description = "The largest number of reviews a single job may carry."
  type        = number
  default     = 1000
}

variable "worker_chunk_size" {
  description = "How many reviews the worker processes per checkpointed chunk."
  type        = number
  default     = 25
}

variable "bert_batch_size" {
  description = "How many sub-comments are labelled by BERT at once."
  type        = number
  default     = 64
}

variable "sqs_visibility_timeout" {
  description = "The SQS visibility timeout in seconds. The worker renews it every heartbeat."
  type        = number
  default     = 600
}

variable "sqs_max_receive_count" {
  description = "How many times a message may be received before it moves to the dead-letter queue."
  type        = number
  default     = 3
}

variable "sqs_message_retention_seconds" {
  description = "How long a job message is kept on the queue."
  type        = number
  default     = 345600
}

variable "heartbeat_interval" {
  description = "How often the worker renews the SQS visibility timeout and the DynamoDB lease, in seconds."
  type        = number
  default     = 60
}

variable "job_lease_seconds" {
  description = "How long a claimed job stays owned before another worker may take it over."
  type        = number
  default     = 600
}

variable "job_ttl_days" {
  description = "How long a job record survives in DynamoDB before its TTL expires it."
  type        = number
  default     = 7
}

variable "job_data_expiration_days" {
  description = "How long job inputs and results survive in S3 before the lifecycle rule expires them."
  type        = number
  default     = 7
}

variable "log_retention_days" {
  description = "How long the CloudWatch log group keeps its events."
  type        = number
  default     = 7
}

variable "log_level" {
  description = "The log level the service runs at."
  type        = string
  default     = "INFO"
}

variable "tags" {
  description = "Tags applied to every taggable resource."
  type        = map(string)
  default = {
    Project = "reviewassist"
    Managed = "terraform"
  }
}
