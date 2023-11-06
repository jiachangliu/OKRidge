import psutil
import math
import requests

bytes_per_GB = 1024**3

def convert_bytes_to_GB(x_bytes):
    x_GB = x_bytes / bytes_per_GB
    return x_GB

def convert_GB_to_bytes(x_GB):
    x_bytes = x_GB * bytes_per_GB
    return x_bytes

def get_RAM_available_in_bytes():
    return psutil.virtual_memory()[1]

def get_RAM_available_in_GB():
    return convert_bytes_to_GB(get_RAM_available_in_bytes())

def get_RAM_used_in_bytes():
    return psutil.virtual_memory()[3]

def get_RAM_used_in_GB():
    return convert_bytes_to_GB(get_RAM_used_in_bytes())

def round_down_n_decimal_places(a, n):
    return math.floor(a * 10**n) / 10**n

def download_file_from_google_drive(id, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # link: https://stackoverflow.com/a/39225272/5040208
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
