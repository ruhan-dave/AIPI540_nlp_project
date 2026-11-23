# The application stack's remote state.
#
# The bucket and key come from `backend.hcl` (created from backend.hcl.example
# and gitignored), so this block stays free of environment-specific values.
# `use_lockfile` gives the S3 backend a native lock file — it needs Terraform
# >= 1.10.

terraform {
  backend "s3" {
    use_lockfile = true
    encrypt      = true
  }
}
