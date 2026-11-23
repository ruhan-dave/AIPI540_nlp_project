output "state_bucket_name" {
  description = "The name of the remote-state bucket."
  value       = aws_s3_bucket.state.id
}

output "state_bucket_arn" {
  description = "The ARN of the remote-state bucket."
  value       = aws_s3_bucket.state.arn
}

output "state_bucket_region" {
  description = "The region the remote-state bucket lives in."
  value       = var.region
}

output "backend_config" {
  description = <<-EOT
    The backend configuration to copy into `infra/bootstrap/backend.hcl` and
    `infra/terraform/backend.hcl`. The two roots use different keys so they
    never share a state file.
  EOT
  value = {
    bucket = aws_s3_bucket.state.id
    key    = "bootstrap/terraform.tfstate"
    region = var.region
  }
}

output "application_backend_config" {
  description = "The backend configuration for the application root (infra/terraform/backend.hcl)."
  value = {
    bucket = aws_s3_bucket.state.id
    key    = "application/terraform.tfstate"
    region = var.region
  }
}
