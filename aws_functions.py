import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os

def check_file_exists_in_s3(file_path):
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    s3_client = boto3.client('s3')
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise e

def download_files_from_s3(local_folder, file_path_list):
    s3 = boto3.client('s3')
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    folder_prefix = ''
    
    try:
        # List objects in the S3 bucket
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)
        
        # Download filtered files
        for page in page_iterator:
            for obj in page.get('Contents', []):
                key = obj['Key']
                
                # Apply file filter if specified
                if key not in file_path_list:
                    continue
                
                # Construct local file path
                local_path = os.path.join(local_folder, key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                try:
                    print(f"Downloading: {key} -> {local_path}")
                    s3.download_file(bucket_name, key, local_path)
                    print(f"Downloaded: {local_path}")
                except Exception as e:
                    print(f"Error downloading {key}: {e}")
    except NoCredentialsError:
        print("No AWS credentials found.")
    except Exception as e:
        print(f"An error occurred: {e}")