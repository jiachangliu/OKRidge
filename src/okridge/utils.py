import psutil
import math

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