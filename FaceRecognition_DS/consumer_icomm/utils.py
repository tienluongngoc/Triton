import os
import datetime


def load_parameter(file_config='config/parameter.cfg'):
    param = {}
    with open(file_config) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            else:
                key = line.split("=")[0]
                val = "=".join(line.split("=")[1:])
                if val.isdigit():
                    val = int(val)
                else:
                    try:
                        val = float(val)
                    except Exception as ve:
                        pass
                param[key] = val
    return param


def gen_folder_name(path_img, hash_url, datetime_obj):
    date_n = datetime_obj.strftime("%Y-%m-%d")
    hour_n = str(datetime_obj.hour)
    path_data = os.path.join(path_img, hash_url, date_n, hour_n)
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    return path_data
