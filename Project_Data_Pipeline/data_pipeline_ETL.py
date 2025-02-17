import os
import logging
import boto3
import pandas as pd
import snowflake.connector
import uuid
from snowflake.connector import connect
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

# Load variables for aws and snowflake from .env file (enter file path if .env file in another place)
load_dotenv("your/file/path.env")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_FILE_KEY = os.getenv("S3_FILE_KEY")
LOCAL_FILE_PATH = os.getenv("LOCAL_FILE_PATH")

# Snowflake Configuration
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_STAGE = os.getenv("SNOWFLAKE_STAGE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_TABLE = os.getenv("SNOWFLAKE_TABLE")
TEMP_CSV_PATH = os.getenv('temp_csv_path')

# Downloading file from AWS S3
def extract_from_s3():
    logging.info("Extracting data from S3...")
    try:
        s3 = boto3.client(
            "s3", 
            region_name="eu-west-1", # in which region you can find the bucket
            aws_access_key_id = AWS_ACCESS_KEY, # using the Access Key provided by env
            aws_secret_access_key = AWS_SECRET_KEY # usign the Secret Key provided by env
        )
        s3.download_file(S3_BUCKET_NAME, S3_FILE_KEY, LOCAL_FILE_PATH)
        logging.info(f"File {S3_FILE_KEY} downloaded from S3 to {LOCAL_FILE_PATH}")
        return LOCAL_FILE_PATH
    except Exception as e:
        logging.error(f"Error in extracting data from S3: {e}")
        raise

# Transforming the data
def transform_data(file_path):
    logging.info("Transforming data...")
    try:
        # Loading data
        data = pd.read_csv(file_path,header=1)
                
        # Renaming unknown column 
        data.rename(columns={'Unnamed: 1':'region_country_area_name'}, inplace=True)
        # Normalizing column names
        data.columns = data.columns.str.lower().str.replace(' ','_').str.replace('/','_')
        
        # replace the commas by '' because snowflake would split the columns at every comma and create more columns 
        data = data.replace({',': ''}, regex=True)
        
        # Handling missing values
        data['last_election_date_footnote'].fillna('No Footnote', inplace=True)
        data['footnotes'].fillna('No Footnote', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)  # Drop columns that are NA
                
        # Converting Year to integer and Value to numeric
        data['year']= pd.to_datetime(data["year"], format = '%Y').dt.year
        data['percentage']= pd.to_numeric(data['value'], errors = 'coerce')
        data['last_election_date'] = pd.to_datetime(data['last_election_date'], format = '%Y-%m')
        
        # Normalizing the value column and saving it to a new column
        scaler = MinMaxScaler()
        data['normalized_value'] = scaler.fit_transform(data[['percentage']])
        
        # Adding processing timestamp
        data['processed_at'] = pd.to_datetime('now')
        
        # creating a UUID column for a unique identifier per row
        data['id'] = [str(uuid.uuid4()) for _ in range (len(data))]
        
        logging.info(f'Data transformation complete. Shape: {data.shape}')
                
        return data
    except Exception as e:
        logging.error(f'Error in transformning data: {e}')
        raise
    
 
# Load transformed data into snowflake   
def load_to_snowflake(data):
    logging.info("Loading data into Snowflake...")
    try:
        conn = snowflake.connector.connect(
            user = SNOWFLAKE_USER,
            password = SNOWFLAKE_PASSWORD,
            account = SNOWFLAKE_ACCOUNT,
            database = SNOWFLAKE_DATABASE,
            schema = SNOWFLAKE_SCHEMA
        )
        cur = conn.cursor()
        
        # Defining the table name
        table_name = SNOWFLAKE_TABLE
        schema_name = SNOWFLAKE_SCHEMA
        
        # Checking if the table exists
        cur.execute(f"""
                    SELECT COUNT (*) FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_NAME = '{table_name.upper()}'
                    AND TABLE_SCHEMA = '{schema_name.upper()}';
                    """)
        table_exists = cur.fetchone()[0] > 0
          
        # Creating the table if it does not exist
        if not table_exists:
            logging.info(f"Table {table_name} does not exist. Creating table...")
            
            # Define the expected columns and their corresponding data types
            col_types = {
                'id': 'STRING',
                'region_country_area': 'STRING',
                'region_country_area_name': 'STRING',
                'year': 'NUMBER(4,0)',
                'series': 'STRING',
                'last_election_date': 'DATE',
                'last_election_date_footnote': 'STRING',
                'percentage': 'NUMBER(38,4)',
                'footnotes': 'STRING',
                'source': 'STRING',
                'normalized_value': 'NUMBER(38,10)',
                'processed_at': 'TIMESTAMP'
            }

            # Generating the CREATE TABLE query based on the dictionary
            create_table_query = f"CREATE TABLE {table_name} ("
            for col, dtype in col_types.items():
                create_table_query += f"{col} {dtype},"
            create_table_query = create_table_query.rstrip(",") + ");"
            
            cur.execute(create_table_query)
            logging.info(f"Table {table_name} created successfully.")
               
        
        # Saving the dataframe temporary to csv and check that only the expected columns are in the dataframe
        expected_columns = ['id','region_country_area', 'region_country_area_name', 'year', 'series', 
                    'last_election_date', 'last_election_date_footnote', 'percentage', 'footnotes', 
                    'source', 'normalized_value', 'processed_at']
        data = data[expected_columns]
        data.to_csv(TEMP_CSV_PATH , index = False, sep=',', quoting=1)   
        
        # uploading the file to Snowflake staging area
        cur.execute(f"PUT file://{TEMP_CSV_PATH} @PARLIAMENT_STAGE")
        logging.info("File uploaded to Snowflake staging area.")
        
        # Loading data into Snowflake table
        cur.execute(f"""
                COPY INTO {table_name}
                FROM @PARLIAMENT_STAGE/{os.path.basename(TEMP_CSV_PATH)}
                FILE_FORMAT = (TYPE = 'CSV', SKIP_HEADER=1,FIELD_OPTIONALLY_ENCLOSED_BY = '"',NULL_IF = (''));
        """)
        logging.info("Data successfully loaded into Snowflake.")
        
        # Cleaning up the staging area
        cur.execute(f"REMOVE @PARLIAMENT_STAGE/{os.path.basename(TEMP_CSV_PATH)}")
        logging.info("Temporary staged file removed.")
        
        # Closing Connections
        cur.close()
        conn.close()
    
    except Exception as e:
        logging.error(f"Error in loading data into Snowflake: {e}")
   
        raise 
 
# Creating main ETL Pipeline   
def main():
    try:
        file_path = extract_from_s3() # Extraction from AWS
        transformed_data = transform_data(file_path) # Data Transformation
        load_to_snowflake(transformed_data) # Data Loading into Snowflake
        logging.info("ETL Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"ETL Pipeline failed: {e}")

# Code is only running if script is directly started
if __name__ == "__main__":
    main()
