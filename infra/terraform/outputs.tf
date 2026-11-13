output "aws_region" {
  description = "The region the stack was created in."
  value       = var.region
}

output "api_ecr_repository_url" {
  description = "The ECR repository to push the backend image to."
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "The ECS cluster the API and the worker run on."
  value       = aws_ecs_cluster.main.name
}

output "api_service_name" {
  description = "The ECS service running the FastAPI server."
  value       = aws_ecs_service.api.name
}

output "worker_service_name" {
  description = "The ECS service running the SQS worker."
  value       = aws_ecs_service.worker.name
}

output "api_url" {
  description = "The public HTTPS URL the frontend calls. Point the API domain at the load balancer's DNS name."
  value       = "https://${var.api_domain_name}"
}

output "alb_dns_name" {
  description = "The load balancer's DNS name — the target of the API domain's A/ALIAS record."
  value       = aws_lb.api.dns_name
}

output "queue_url" {
  description = "The SQS queue that carries job references."
  value       = aws_sqs_queue.jobs.url
}

output "dlq_url" {
  description = "The dead-letter queue for jobs that failed three times."
  value       = aws_sqs_queue.dlq.url
}

output "jobs_table_name" {
  description = "The DynamoDB table holding the job records."
  value       = aws_dynamodb_table.jobs.name
}

output "job_data_bucket" {
  description = "The private S3 bucket holding job inputs and results."
  value       = aws_s3_bucket.job_data.id
}

output "log_group_name" {
  description = "The CloudWatch log group the API and the worker write to."
  value       = aws_cloudwatch_log_group.backend.name
}

output "api_secret_arn" {
  description = "The secret holding the API bearer token. Its value is seeded out of band."
  value       = local.api_secret_arn
}

output "provider_secret_arn" {
  description = "The secret holding the LLM provider key. Its value is seeded out of band."
  value       = local.provider_secret_arn
}
