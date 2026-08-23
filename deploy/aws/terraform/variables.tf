variable "project" {
  type    = string
  default = "appraisenet"
}

variable "environment" {
  type    = string
  default = "staging"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "vpc_id" {
  description = "VPC to deploy into"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnets for the load balancer (two or more AZs)"
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Private subnets for the ECS tasks and the database (NAT access required for pip/boto in the MLflow sidecar)"
  type        = list(string)
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS on the load balancer; empty = HTTP only"
  type        = string
  default     = ""
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "mlflow_image" {
  type    = string
  default = "ghcr.io/mlflow/mlflow:v3.4.0"
}

variable "container_port" {
  type    = number
  default = 5000
}

variable "task_cpu" {
  type    = number
  default = 1024
}

variable "task_memory" {
  type    = number
  default = 2048
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "max_count" {
  type    = number
  default = 4
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "postgres_version" {
  type    = string
  default = "16"
}

variable "db_name" {
  type    = string
  default = "appraisenet"
}

variable "db_username" {
  type    = string
  default = "appraisenet"
}
