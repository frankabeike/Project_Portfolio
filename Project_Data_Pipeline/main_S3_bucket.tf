terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Define variables for secure access by passing credentials at runtime
variable "aws_access_key" {}
variable "aws_secret_key" {}

# select aws as the provider
provider "aws" {
  region = "eu-west-1" # specified desired region
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
}

# create the S3 bucket
resource "aws_s3_bucket" "undatabucket" {
  bucket = "undatabucketfb"
}

# upload the CSV File to S3
resource "aws_s3_object" "undatacsv" {
  bucket = aws_s3_bucket.undatabucket.bucket # Referenced the bucket created earlier
  key = "data/seats_held_by_women.csv" # Path and Name of the file within the S3 bucket
  source = "your/file/path.csv" # Patch to the local CSV file
  acl = "public-read" # Permissions for the file
}

