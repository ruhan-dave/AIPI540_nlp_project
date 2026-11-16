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

variable "alb_subnet_ids" {
  description = "The subnets the internal load balancer sits in. Empty means `public_subnet_ids`: the load balancer is internal, so the subnets only have to be routable to the VPC link's network interfaces."
  type        = list(string)
  default     = []
}

variable "vpc_link_subnet_ids" {
  description = "The subnets the API Gateway VPC link places its network interfaces in. Empty means `public_subnet_ids`. The link must reach the load balancer, so these subnets have to be routable to the load balancer's subnets."
  type        = list(string)
  default     = []
}

variable "vpc_link_security_group_ids" {
  description = "Security groups for the API Gateway VPC link's network interfaces. Empty means the stack creates one that can reach the load balancer over private HTTP."
  type        = list(string)
  default     = []
}

variable "api_image_tag" {
  description = "The tag of the backend image in the ECR repository. The image must be pushed before the services are started."
  type        = string
  default     = "v1"
}

variable "api_domain_name" {
  description = "REMOVED — the API Gateway HTTP API's generated execute-api endpoint is the public URL, so no domain and no ACM certificate are needed. Kept as a deprecated no-op so a stale `terraform.tfvars` naming it is rejected rather than silently ignored."
  type        = string
  default     = null

  validation {
    condition     = var.api_domain_name == null
    error_message = "api_domain_name is no longer used: the public endpoint is the API Gateway's generated execute-api URL. Remove it from terraform.tfvars."
  }
}

variable "acm_certificate_arn" {
  description = "REMOVED — the public HTTPS endpoint is the API Gateway's generated execute-api address, which comes with its own certificate. Kept as a deprecated no-op so a stale `terraform.tfvars` naming it is rejected rather than silently ignored."
  type        = string
  default     = null

  validation {
    condition     = var.acm_certificate_arn == null
    error_message = "acm_certificate_arn is no longer used: the load balancer is internal and the public endpoint is the API Gateway's generated execute-api URL. Remove it from terraform.tfvars."
  }
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
  description = "The largest CSV a single job may carry (50 MiB). This is the limit for the presigned S3 upload path, where the CSV never passes through the API Gateway."
  type        = number
  default     = 52428800
}

variable "max_multipart_upload_bytes" {
  description = "The largest CSV the multipart `POST /jobs` route accepts (8 MiB). An API Gateway HTTP API caps a request payload at 10 MB and that limit cannot be raised, so the multipart path is sized to leave room for the multipart framing."
  type        = number
  default     = 8388608
}

variable "upload_url_ttl_seconds" {
  description = "How long a presigned upload URL stays valid."
  type        = number
  default     = 900
}

variable "predict_timeout_seconds" {
  description = "The budget for one LLM split request, provider retries included. A split that overruns it falls back to the whole review."
  type        = number
  default     = 20

  validation {
    condition     = var.predict_timeout_seconds < var.predict_deadline_seconds
    error_message = "predict_timeout_seconds must be shorter than predict_deadline_seconds, or the split could exhaust the whole prediction budget on its own."
  }
}

variable "worker_split_timeout_seconds" {
  description = "The budget for one LLM split in the worker. The worker has no gateway in front of it and can afford to wait longer than the API's budget for one review's split."
  type        = number
  default     = 60
}

variable "predict_deadline_seconds" {
  description = "The total budget for one `POST /predict`, split and BERT included. A request that overruns it returns the whole-review fallback instead of being cut off by the API Gateway."
  type        = number
  default     = 25

  validation {
    condition     = var.predict_deadline_seconds < floor(var.http_api_integration_timeout_ms / 1000)
    error_message = "predict_deadline_seconds must be shorter than the API Gateway integration timeout, or the gateway would answer 504 while the API was still working."
  }
}

variable "http_api_integration_timeout_ms" {
  description = "The API Gateway integration timeout in milliseconds. The HTTP API maximum is 30000 and it cannot be raised, so the service's prediction deadline has to fit inside it."
  type        = number
  default     = 29000

  validation {
    condition     = var.http_api_integration_timeout_ms > 0 && var.http_api_integration_timeout_ms <= 30000
    error_message = "http_api_integration_timeout_ms must be between 1 and 30000 — an HTTP API's integration timeout cannot exceed 30 seconds."
  }
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
