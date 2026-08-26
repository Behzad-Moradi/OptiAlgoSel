import json
import boto3
from botocore.exceptions import ClientError

secrets_client = boto3.client("secretsmanager")

def get_rds_credentials():
    try:
        response = secrets_client.get_secret_value(SecretId="optialgosel/rds/credentials")
        return json.loads(response["SecretString"])
    except ClientError as e:
        print(f"Error retrieving secret: {e.response['Error']['Message']}")
        raise e
