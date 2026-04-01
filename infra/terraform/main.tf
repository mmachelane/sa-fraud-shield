terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }

  # Remote state — uncomment when deploying
  # backend "s3" {
  #   bucket         = "sa-fraud-shield-tfstate"
  #   key            = "prod/terraform.tfstate"
  #   region         = "af-south-1"
  #   dynamodb_table = "sa-fraud-shield-tflock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "sa-fraud-shield"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
