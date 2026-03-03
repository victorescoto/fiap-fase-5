output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "service_url" {
  value = try(aws_apprunner_service.api[0].service_url, "")
}

output "dashboard_ecr_repository_url" {
  value = aws_ecr_repository.dashboard.repository_url
}

output "dashboard_url" {
  value = try(aws_apprunner_service.dashboard[0].service_url, "")
}