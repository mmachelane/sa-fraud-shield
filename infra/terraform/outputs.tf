output "api_endpoint" {
  description = "ALB DNS name for the fraud-shield API"
  value       = "https://${aws_lb.api.dns_name}"
}

output "kafka_bootstrap_brokers_tls" {
  description = "MSK TLS bootstrap brokers"
  value       = aws_msk_cluster.main.bootstrap_brokers_tls
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = "${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379"
}

output "mlflow_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow.bucket
}

output "data_bucket" {
  description = "S3 bucket for fraud data and model artifacts"
  value       = aws_s3_bucket.data.bucket
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}
