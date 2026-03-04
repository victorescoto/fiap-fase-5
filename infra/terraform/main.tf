provider "aws" {
  region = var.region
}

# ----------------------------
# ECR Repositories
# ----------------------------
resource "aws_ecr_repository" "api" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "dashboard" {
  name                 = var.dashboard_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ----------------------------
# IAM: App Runner access to ECR
# ----------------------------
resource "aws_iam_role" "apprunner_ecr_access" {
  name = "${var.project_name}-apprunner-ecr-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# ----------------------------
# App Runner: API (FastAPI)
# ----------------------------
resource "aws_apprunner_service" "api" {
  count        = var.enable_apprunner ? 1 : 0
  service_name = var.project_name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }

    image_repository {
      image_configuration {
        port = "8000"
      }

      image_identifier      = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
    }

    auto_deployments_enabled = true
  }

  depends_on = [aws_iam_role_policy_attachment.apprunner_ecr_access]
}