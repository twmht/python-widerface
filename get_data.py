import requests
import os
import zipfile


data_dir = 'data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

image_zip_file = 'WIDER_train.zip'
annotation_url = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip'


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_file_from_web_server(url, destination):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    response = requests.get(url, stream=True)
    save_response_content(response, os.path.join(destination, local_filename))

    return local_filename


#  TODO Add progress bar
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def extract_zip_file(zip_file_name, destination):
    zip_ref = zipfile.ZipFile(zip_file_name, 'r')
    zip_ref.extractall(destination)
    zip_ref.close()


if __name__ == "__main__":
    filename = 'WIDER_train.zip'
    file_id = '0B6eKvaijfFUDQUUwd21EckhUbWs'
    destination = 'data/WIDER_train.zip'

    print('downloading the images from google drive...')
    download_file_from_google_drive(file_id, destination)

    extract_zip_file(os.path.join(data_dir, image_zip_file), data_dir)

    print('downloading the bounding boxes annotations...')
    annotation_zip_file = download_file_from_web_server(annotation_url,
                                                        data_dir)
    extract_zip_file(os.path.join(data_dir, annotation_zip_file), data_dir)

    print('done')
