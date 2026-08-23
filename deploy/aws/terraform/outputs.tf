output "api_url" {
  value = var.certificate_arn == "" ? "http://${aws_lb.this.dns_name}" : "https://${aws_lb.this.dns_name}"
}

output "ecr_repository" {
  value = aws_ecr_repository.api.repository_url
}

output "ecs_cluster" {
  value = aws_ecs_cluster.this.name
}

output "ecs_service" {
  value = aws_ecs_service.api.name
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "database_endpoint" {
  value     = aws_db_instance.postgres.address
  sensitive = true
}

output "database_url_parameter" {
  description = "SSM parameter holding DATABASE_URL (SecureString)"
  value       = aws_ssm_parameter.database_url.name
}
