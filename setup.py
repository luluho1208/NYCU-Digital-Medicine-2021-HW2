from google_drive_downloader import GoogleDriveDownloader as gdd
import shutil
import wget
import os


############################# Downlaod train data ############################
try:
    shutil.rmtree("./train_data")
except OSError as e:
    print(e)
else:
    print("The train data directory is deleted successfully")

gdd.download_file_from_google_drive(file_id='18NjHL_8LiemFvzfPf_1DzfjUfwzYSf4g', dest_path='./train_data.zip', unzip=True)
os.remove("./train_data.zip")
print("The train data has been downloaded successfully")

############################# Downlaod test data #############################

try:
    shutil.rmtree("./test_data")
except OSError as e:
    print(e)
else:
    print("The test data directory is deleted successfully")

gdd.download_file_from_google_drive(file_id='1FhwKW425FUQaTBogviyFJGNt7AK4KLOp', dest_path='./test_data.zip', unzip=True)
os.remove("./test_data.zip")
print("The test data has been downloaded successfully")


######################## Downlaod ViT Pretrain Weight ########################
site_url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
file_name = wget.download(site_url, out= "./pretrain_weight")
print(file_name)
print("The ViT pretrain weight has been downloaded successfully")