data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

resource "random_id" "state_bucket_suffix" {
  byte_length = 4
}

locals {
  state_bucket_name = coalesce(
    var.state_bucket_name,
    "${var.name_prefix}-tfstate-${random_id.state_bucket_suffix.hex}",
  )
}

# The state bucket. It holds the state of both Terraform roots, so it must
# outlive the application stack: destroying it is a separate, explicitly
# approved teardown step.
resource "aws_s3_bucket" "state" {
  bucket = local.state_bucket_name

  # Never silently erase state history. The teardown checklist empties the
  # bucket explicitly, version by version, before this is destroyed.
  force_destroy = false

  tags = merge(var.tags, { Name = local.state_bucket_name })

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket = aws_s3_bucket.state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_iam_policy_document" "state" {
  # State is read and written over TLS or not at all.
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }

  # Optional: restrict state access to the deployment role. Off by default so
  # the bootstrap cannot lock the operator out of their own state.
  dynamic "statement" {
    for_each = length(var.allowed_principals) > 0 ? [true] : []

    content {
      sid       = "DenyOutsideDeploymentRole"
      effect    = "Deny"
      actions   = ["s3:*"]
      resources = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]

      principals {
        type        = "*"
        identifiers = ["*"]
      }

      condition {
        test     = "StringNotEquals"
        variable = "aws:PrincipalArn"
        values   = var.allowed_principals
      }
    }
  }
}

resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id
  policy = data.aws_iam_policy_document.state.json
}
