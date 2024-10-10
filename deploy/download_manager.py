import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "deploy/key.json"

def is_local_folder_empty(local_folder:str) -> int:
    return len(os.listdir(local_folder)) == 0


def download_bucket_content(bucket_name, local_folder):
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    if is_local_folder_empty(local_folder):
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        blobs = bucket.list_blobs()
        for blob in blobs:
            print(blob)
            if blob.name.endswith('/'):  
                continue
            local_file_path = os.path.join(local_folder, blob.name.split('/')[-1])
            blob.download_to_filename(local_file_path)
            print(f"Downloaded: {blob.name}")
    else:
        print("Content already downloaded")


