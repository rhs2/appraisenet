# Copy to prod.tfvars (git-ignored) and fill in your account's values.
environment        = "prod"
region             = "us-east-1"
vpc_id             = "vpc-xxxxxxxx"
public_subnet_ids  = ["subnet-aaaaaaaa", "subnet-bbbbbbbb"]
private_subnet_ids = ["subnet-cccccccc", "subnet-dddddddd"]
certificate_arn    = ""          # ACM certificate ARN for HTTPS, optional
image_tag          = "latest"
desired_count      = 2
db_instance_class  = "db.t4g.small"
