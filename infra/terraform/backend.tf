terraform {
  backend "s3" {
    bucket         = "fiap-fase-5-tfstate-583260078901"
    key            = "passos-magicos-api/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "fiap-fase-5-terraform-lock"
    encrypt        = true
  }
}