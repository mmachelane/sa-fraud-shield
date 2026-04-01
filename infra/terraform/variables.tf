variable "aws_region" {
  description = "AWS region — af-south-1 (Cape Town) for SA data residency (POPIA compliance)"
  type        = string
  default     = "af-south-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones — af-south-1 has 3 AZs"
  type        = list(string)
  default     = ["af-south-1a", "af-south-1b", "af-south-1c"]
}

variable "private_subnet_cidrs" {
  description = "CIDRs for private subnets (ECS, MSK, ElastiCache)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDRs for public subnets (ALB)"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# ── ECS ───────────────────────────────────────────────────────────────────────
variable "api_image" {
  description = "Docker image URI for the fraud-shield API (ECR)"
  type        = string
  default     = "123456789012.dkr.ecr.af-south-1.amazonaws.com/sa-fraud-shield:latest"
}

variable "api_cpu" {
  description = "ECS task CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "api_memory" {
  description = "ECS task memory in MiB"
  type        = number
  default     = 2048
}

variable "api_desired_count" {
  description = "Number of ECS task replicas"
  type        = number
  default     = 2
}

# ── MSK ───────────────────────────────────────────────────────────────────────
variable "msk_instance_type" {
  description = "MSK broker instance type"
  type        = string
  default     = "kafka.t3.small"
}

variable "msk_broker_count" {
  description = "Number of MSK brokers (must be multiple of AZ count)"
  type        = number
  default     = 3
}

variable "msk_kafka_version" {
  description = "Apache Kafka version"
  type        = string
  default     = "3.6.0"
}

variable "msk_volume_size" {
  description = "EBS volume size per broker in GiB"
  type        = number
  default     = 100
}

# ── ElastiCache ───────────────────────────────────────────────────────────────
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 1
}

# ── S3 ────────────────────────────────────────────────────────────────────────
variable "mlflow_bucket_name" {
  description = "S3 bucket for MLflow artifacts"
  type        = string
  default     = "sa-fraud-shield-mlflow-artifacts"
}

variable "data_bucket_name" {
  description = "S3 bucket for fraud shield data (parquet, models)"
  type        = string
  default     = "sa-fraud-shield-data"
}
