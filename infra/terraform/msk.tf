# ── MSK Cluster (Managed Kafka) ───────────────────────────────────────────────
resource "aws_msk_cluster" "main" {
  cluster_name           = "${local.name_prefix}-kafka"
  kafka_version          = var.msk_kafka_version
  number_of_broker_nodes = var.msk_broker_count

  broker_node_group_info {
    instance_type   = var.msk_instance_type
    client_subnets  = aws_subnet.private[*].id
    security_groups = [aws_security_group.msk.id]

    storage_info {
      ebs_storage_info {
        volume_size = var.msk_volume_size
      }
    }
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS_PLAINTEXT"
      in_cluster    = true
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
    }
  }

  tags = { Name = "${local.name_prefix}-kafka" }
}

resource "aws_msk_configuration" "main" {
  name              = "${local.name_prefix}-kafka-config"
  kafka_versions    = [var.msk_kafka_version]
  description       = "sa-fraud-shield Kafka configuration"

  server_properties = <<-EOT
    auto.create.topics.enable=false
    default.replication.factor=3
    min.insync.replicas=2
    num.partitions=12
    log.retention.hours=168
    log.retention.bytes=107374182400
  EOT
}

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${local.name_prefix}"
  retention_in_days = 30
}

# ── MSK Topics (via separate app bootstrap — MSK doesn't support topic resources)
# Topics created by kafka-init container at app startup:
#   transactions    — 12 partitions, 7d retention
#   fraud-alerts    — 3 partitions, 30d retention
#   feature-updates — 6 partitions, 1d retention
