import shutil
from google.cloud import storage
import os


def create_model_zip():
    shutil.make_archive("./src/saved_models", 'zip',"./src/saved_models/" )
def create_log_zip():
    shutil.make_archive("./src/lightning_logs", 'zip',"./lightning_logs/" )

def upload_model_file(file_path,version):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket('qgen_models')
    blob=bucket.blob(f"model/saved_models{version}.zip")
    with open(file_path,'rb') as f:
        blob.upload_from_file(f)

def download_model_file(download_path,version):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket('qgen_models')
    blob=bucket.blob(f"model/saved_models{version}.zip")
    blob.download_to_filename(download_path)

def upload_log_file(file_path,version):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket('qgen_models')
    blob=bucket.blob(f"model/lightning_logs{version}.zip")
    with open(file_path,'rb') as f:
        blob.upload_from_file(f)

def download_log_file(download_path,version):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket('qgen_models')
    blob=bucket.blob(f"model/lightning_logs{version}.zip")
    blob.download_to_filename(download_path)


def get_latest_version():
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket('qgen_models')
    blobs = list(bucket.list_blobs())
    names = list(filter(lambda x: "model/saved_models" in x,[i.name for i in blobs]))
    if names==[]:
        return 0
    else:
        return max([int(i[18:-4]) for i in names])

def delete_zips():
    os.remove("./src/lightning_logs.zip")
    os.remove("./src/saved_models.zip")

def download_zips():
    model_zip="/root/Question_generation/src/saved_models.zip"
    log_zip="/root/Question_generation/src/lightning_logs.zip"
    version=get_latest_version()
    download_log_file(log_zip,version)
    download_model_file(model_zip,version)

def sync_upload():
    print("Zipping Files")
    create_log_zip()
    print("Logs Zipped")
    create_model_zip()
    print("Models Zipped")
    version=get_latest_version()
    model_zip="/root/Question_generation/src/saved_models.zip"
    log_zip="/root/Question_generation/src/lightning_logs.zip"
    print("Uploading logs")
    upload_log_file(log_zip,version+1)
    print("Uploading models")
    upload_model_file(model_zip,version+1)
    print("deleting_zips")
    delete_zips()



def sync_download():
    print("Downloading zip")
    download_zips()
    print("Downloaded zips")
    print("unpacking")
    shutil.rmtree("./src/saved_models/")
    os.mkdir("./src/saved_models/")
    shutil.rmtree("./lightning_logs/")
    os.mkdir("./lightning_logs/")
    shutil.unpack_archive("./src/saved_models.zip",extract_dir="./src/saved_models/")
    shutil.unpack_archive("./src/lightning_logs.zip",extract_dir="./lightning_logs/")
    delete_zips()


sync_download()


