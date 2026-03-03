output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "service_url" {
  value = try(aws_apprunner_service.api[0].service_url, "")
}