# The ReviewAssist AWS stack: the API, the worker and the job infrastructure.
#
# What this stack creates:
#   * one ECR repository for the backend image
#   * one ECS cluster with two Fargate services (the API and the worker) that
#     share the image but have separate task definitions
#   * one SQS queue with a dead-letter queue
#   * one on-demand DynamoDB table for job records, with a TTL
#   * one private S3 bucket for job inputs and results
#   * two Secrets Manager secrets (metadata only — the values are seeded by hand)
#   * one CloudWatch log group
#   * an ALB with an HTTPS listener and an ACM certificate
#   * the IAM roles and security groups those need
#
# What it deliberately does NOT create: a VPC, subnets, a NAT gateway, VPC
# endpoints, API Gateway, WAF, ElastiCache, autoscaling or Container Insights.
# The tasks get public IPs in the existing public subnets instead.

data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

resource "random_id" "job_bucket" {
  byte_length = 4
}

locals {
  job_bucket_name = "${var.name_prefix}-jobs-${random_id.job_bucket.hex}"

  # The provider key is read from Secrets Manager under the name that matches
  # the configured provider.
  provider_secret_env = var.llm_provider == "gemini" ? "GEMINI_API_KEY" : "OPENROUTER_API_KEY"

  # Terraform manages the secrets' metadata and ARNs. The values are seeded out
  # of band, so they never enter the Terraform state.
  api_secret_arn      = coalesce(var.api_secret_arn, aws_secretsmanager_secret.api.arn)
  provider_secret_arn = coalesce(var.provider_secret_arn, aws_secretsmanager_secret.provider.arn)

  tags = merge(var.tags, { Name = var.name_prefix })
}

# --- ECR ------------------------------------------------------------------

resource "aws_ecr_repository" "api" {
  name = "${var.name_prefix}-api"

  # The demo repository's pushed image manifests go with it, so `terraform
  # destroy` leaves no orphaned images behind.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = local.tags
}

# --- ECS ------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = "${var.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled"
  }

  tags = local.tags
}

# --- CloudWatch -----------------------------------------------------------

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${var.name_prefix}"
  retention_in_days = var.log_retention_days

  tags = local.tags
}

# --- SQS ------------------------------------------------------------------

resource "aws_sqs_queue" "dlq" {
  name                      = "${var.name_prefix}-jobs-dlq"
  message_retention_seconds = var.sqs_message_retention_seconds
  sqs_managed_sse_enabled   = true

  tags = local.tags
}

resource "aws_sqs_queue" "jobs" {
  name                       = "${var.name_prefix}-jobs"
  visibility_timeout_seconds = var.sqs_visibility_timeout
  message_retention_seconds  = var.sqs_message_retention_seconds
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = var.sqs_max_receive_count
  })

  tags = local.tags
}

data "aws_iam_policy_document" "sqs" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["sqs:*"]
    resources = [aws_sqs_queue.jobs.arn, aws_sqs_queue.dlq.arn]

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
}

resource "aws_sqs_queue_policy" "jobs" {
  queue_url = aws_sqs_queue.jobs.id
  policy    = data.aws_iam_policy_document.sqs.json
}

# --- DynamoDB -------------------------------------------------------------

resource "aws_dynamodb_table" "jobs" {
  name         = "${var.name_prefix}-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  # Job records are disposable: the TTL expires them a week after creation.
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = false
  }

  tags = local.tags
}

# --- S3 job data ----------------------------------------------------------

resource "aws_s3_bucket" "job_data" {
  bucket        = local.job_bucket_name
  force_destroy = false

  tags = local.tags
}

resource "aws_s3_bucket_versioning" "job_data" {
  bucket = aws_s3_bucket.job_data.id
  versioning_configuration {
    # Unversioned by design: the lifecycle rule below expires job data, and a
    # versioned bucket would keep deleted objects alive.
    status = "Suspended"
  }
}

resource "aws_s3_bucket_public_access_block" "job_data" {
  bucket = aws_s3_bucket.job_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "job_data" {
  bucket = aws_s3_bucket.job_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "job_data" {
  bucket = aws_s3_bucket.job_data.id

  rule {
    id     = "expire-job-data"
    status = "Enabled"

    filter {}

    expiration {
      days = var.job_data_expiration_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

data "aws_iam_policy_document" "job_data_bucket" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.job_data.arn, "${aws_s3_bucket.job_data.arn}/*"]

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
}

resource "aws_s3_bucket_policy" "job_data" {
  bucket = aws_s3_bucket.job_data.id
  policy = data.aws_iam_policy_document.job_data_bucket.json
}

# --- Secrets Manager ------------------------------------------------------
#
# Terraform manages the secret metadata and its ARN only.  The values are
# entered in the AWS console (or another out-of-band channel) after the first
# apply, so they never appear in the Terraform state.

resource "aws_secretsmanager_secret" "api" {
  name        = "${var.name_prefix}/api-token"
  description = "Bearer token the frontend presents to the ReviewAssist API."

  # Demo-only secrets: no recovery window, so the approved teardown deletes
  # them for good.  Never apply this to a shared or production secret.
  recovery_window_in_days = 0

  tags = local.tags
}

resource "aws_secretsmanager_secret" "provider" {
  name        = "${var.name_prefix}/${local.provider_secret_env}"
  description = "The ${var.llm_provider} API key used to split reviews."

  recovery_window_in_days = 0

  tags = local.tags
}

# --- Security groups ------------------------------------------------------

resource "aws_security_group" "alb" {
  name        = "${var.name_prefix}-alb"
  description = "Public HTTPS ingress for the ReviewAssist API."
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTPS from the internet (Railway calls the API from outside the VPC)."
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "HTTP to the API tasks."
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_security_group" "api" {
  name        = "${var.name_prefix}-api"
  description = "The API task. Reachable only from the load balancer."
  vpc_id      = var.vpc_id

  ingress {
    description     = "API traffic from the load balancer only."
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Outbound to ECR, CloudWatch, SQS, DynamoDB, S3 and the LLM provider."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_security_group" "worker" {
  name        = "${var.name_prefix}-worker"
  description = "The worker task. No inbound access at all."
  vpc_id      = var.vpc_id

  egress {
    description = "Outbound to SQS, S3, DynamoDB, CloudWatch and the LLM provider."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

# --- ALB ------------------------------------------------------------------

resource "aws_lb" "api" {
  name               = substr("${var.name_prefix}-api", 0, 32)
  load_balancer_type = "application"
  internal           = false

  subnets         = var.public_subnet_ids
  security_groups = [aws_security_group.alb.id]

  idle_timeout               = 120
  drop_invalid_header_fields = true

  # The demo stack is torn down with `terraform destroy`; deletion protection
  # would block the approved teardown.
  enable_deletion_protection = false

  tags = local.tags
}

resource "aws_lb_target_group" "api" {
  name        = substr("${var.name_prefix}-api", 0, 32)
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  deregistration_delay = 30

  # /healthz answers 503 until the local checkpoint is loaded, so the target
  # only starts receiving traffic once the model is ready.
  health_check {
    enabled             = true
    path                = "/healthz"
    matcher             = "200"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  tags = local.tags
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# --- IAM ------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name_prefix}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json

  tags = local.tags
}

resource "aws_iam_role" "task" {
  name               = "${var.name_prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json

  tags = local.tags
}

# The execution role pulls the image, writes the logs and reads the secrets.
resource "aws_iam_role_policy" "execution_ecr" {
  name = "ecr-pull"
  role = aws_iam_role.execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
        ]
        Resource = aws_ecr_repository.api.arn
      },
    ]
  })
}

resource "aws_iam_role_policy" "execution_logs" {
  name = "logs"
  role = aws_iam_role.execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.backend.arn}:*"
      },
    ]
  })
}

resource "aws_iam_role_policy" "execution_secrets" {
  name = "secrets"
  role = aws_iam_role.execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [local.api_secret_arn, local.provider_secret_arn]
      },
    ]
  })
}

data "aws_kms_alias" "secretsmanager" {
  name = "alias/aws/secretsmanager"
}

resource "aws_iam_role_policy" "execution_kms" {
  name = "kms-decrypt"
  role = aws_iam_role.execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = data.aws_kms_alias.secretsmanager.target_key_arn
      },
    ]
  })
}

# The task role is what the running service uses: it reaches the queue, the
# job table and the job-data bucket, and nothing else.
resource "aws_iam_role_policy" "task_sqs" {
  name = "jobs-queue"
  role = aws_iam_role.task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:ChangeMessageVisibility",
          "sqs:SendMessage",
        ]
        Resource = [aws_sqs_queue.jobs.arn, aws_sqs_queue.dlq.arn]
      },
    ]
  })
}

resource "aws_iam_role_policy" "task_dynamodb" {
  name = "jobs-table"
  role = aws_iam_role.task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
        ]
        Resource = aws_dynamodb_table.jobs.arn
      },
    ]
  })
}

resource "aws_iam_role_policy" "task_s3" {
  name = "job-data"
  role = aws_iam_role.task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ]
        Resource = [aws_s3_bucket.job_data.arn, "${aws_s3_bucket.job_data.arn}/*"]
      },
    ]
  })
}

# --- Task definitions -----------------------------------------------------

locals {
  common_environment = [
    { name = "AWS_REGION", value = var.region },
    { name = "JOB_BUCKET", value = aws_s3_bucket.job_data.id },
    { name = "QUEUE_URL", value = aws_sqs_queue.jobs.url },
    { name = "DLQ_URL", value = aws_sqs_queue.dlq.url },
    { name = "JOBS_TABLE", value = aws_dynamodb_table.jobs.name },
    { name = "LLM_PROVIDER", value = var.llm_provider },
    { name = "MODEL_DIR", value = "/app/models/saved_bert_model" },
    { name = "MODEL_DEVICE", value = "cpu" },
    { name = "MAX_UPLOAD_BYTES", value = tostring(var.max_upload_bytes) },
    { name = "MAX_REVIEWS_PER_JOB", value = tostring(var.max_reviews_per_job) },
    { name = "WORKER_CHUNK_SIZE", value = tostring(var.worker_chunk_size) },
    { name = "BERT_BATCH_SIZE", value = tostring(var.bert_batch_size) },
    { name = "SQS_VISIBILITY_TIMEOUT", value = tostring(var.sqs_visibility_timeout) },
    { name = "HEARTBEAT_INTERVAL", value = tostring(var.heartbeat_interval) },
    { name = "JOB_LEASE_SECONDS", value = tostring(var.job_lease_seconds) },
    { name = "JOB_TTL_DAYS", value = tostring(var.job_ttl_days) },
    { name = "RESULT_URL_TTL_SECONDS", value = "900" },
    { name = "LOG_LEVEL", value = var.log_level },
  ]

  # The two secret values never appear as plain environment entries: the task
  # definition refers to them by ARN.
  common_secrets = [
    { name = "API_TOKEN", valueFrom = local.api_secret_arn },
    { name = local.provider_secret_env, valueFrom = local.provider_secret_arn },
  ]

  container_log_configuration = {
    logDriver = "awslogs"
    options = {
      "awslogs-group"         = aws_cloudwatch_log_group.backend.name
      "awslogs-region"        = var.region
      "awslogs-stream-prefix" = "backend"
    }
  }

  health_check = {
    command = [
      "CMD-SHELL",
      "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status == 200 else 1)\"",
    ]
    interval    = 60
    timeout     = 15
    retries     = 3
    startPeriod = 180
  }
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name_prefix}-api"
  cpu                      = "1024"
  memory                   = "2048"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"

  execution_role_arn = aws_iam_role.execution.arn
  task_role_arn      = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        },
      ]

      environment      = local.common_environment
      secrets          = local.common_secrets
      healthCheck      = local.health_check
      logConfiguration = local.container_log_configuration
    },
  ])

  tags = local.tags
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${var.name_prefix}-worker"
  cpu                      = "1024"
  memory                   = "2048"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"

  execution_role_arn = aws_iam_role.execution.arn
  task_role_arn      = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
      essential = true

      command = ["python", "-m", "service.worker"]

      environment = local.common_environment
      secrets     = local.common_secrets
      logConfiguration = merge(local.container_log_configuration, {
        options = merge(local.container_log_configuration.options, {
          "awslogs-stream-prefix" = "worker"
        })
      })
    },
  ])

  tags = local.tags
}

# --- Services -------------------------------------------------------------

resource "aws_ecs_service" "api" {
  name             = "${var.name_prefix}-api"
  cluster          = aws_ecs_cluster.main.id
  task_definition  = aws_ecs_task_definition.api.arn
  desired_count    = var.desired_count
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  # The model loads before the task reports healthy.
  health_check_grace_period_seconds = 240

  network_configuration {
    subnets          = var.public_subnet_ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.https]

  tags = local.tags
}

resource "aws_ecs_service" "worker" {
  name             = "${var.name_prefix}-worker"
  cluster          = aws_ecs_cluster.main.id
  task_definition  = aws_ecs_task_definition.worker.arn
  desired_count    = var.desired_count
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  health_check_grace_period_seconds = 240

  network_configuration {
    subnets          = var.public_subnet_ids
    security_groups  = [aws_security_group.worker.id]
    assign_public_ip = true
  }

  tags = local.tags
}
