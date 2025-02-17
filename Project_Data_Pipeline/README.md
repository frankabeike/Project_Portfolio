# Project Data Pipeline

This repository contains a data pipeline that automates data handling tasks between AWS S3 and Snowflake using Terraform and Python. The project is divided into two main components:

1. **Terraform Script**  
   Responsible for uploading and saving data in an AWS S3 bucket.
   
2. **Python Script**  
   Implements an ETL pipeline to extract data from AWS, transform it, and load it into Snowflake.

## Project Structure

- **terraform/**
  - Contains the Terraform script `main_S3_bucket.tf` used to create and manage AWS S3 bucket resources.
  
- **pipeline/**
  - `data_pipeline_ETL.py`: Python script that executes the ETL process.
  - `your_credentials.env`: Environment file that stores credentials and configurations.  
    **Note:** The `your_credentials.env` file is included in the repository but without my credentials for security reasons. Please replace the placeholders with your own credentials.

## Setup and Configuration

### 1. Terraform Setup

- Navigate to the `terraform/` directory.
- Update the Terraform configuration as needed for your AWS setup.
- Initialize and apply the Terraform script:
  ```bash
  terraform init
  terraform plan
  terraform apply -var "aws_access_key = [your-access-key]" -var "aws_secret_key = [your-secret-key]"
- This will create the required S3 bucket and associated resources

### 2. Environment File (.env)

- In the pipeline/ directory, use the .env file.
- Make sure to replace the placeholder values with your actual AWS and Snowflake credentials.
- Local and Temporary File Path: In the .env file, update the local and temporary file path setting to point to the correct location on your system.

### 3. Python Script Configuration

- Open `data_pipeline_ETL.py`.
- At line 13, update the path to point to the location of your .env file.
- Ensure all credentials and file paths referenced in the script match your .env file.

### 4. Running the Pipeline

- Once the Terraform setup is complete and the .env file is configured, run the Python script:
  ```bash
    python data_pipeline_ETL.py
- The script will:
  1. Extract data from AWS S3.
  2. Transform the data as needed.
  3. Load the transformed data into Snowflake.

### Dependencies
- Terraform: Ensure Terraform is installed and configured on your machine.
- Python: Python 3.12.0 or higher is required.
- Python Packages:
  - Install the necessary Python packages using:
  ```bash
  pip install -r requirements.txt
