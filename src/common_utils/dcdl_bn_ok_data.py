#!/usr/bin/env python3
import csv
import datetime
from dateutil.relativedelta import relativedelta
import requests
from requests.auth import HTTPBasicAuth
import os
import hmac
import math
import base64
import struct
import hashlib
import time
from contextlib import closing
from zipfile import ZipFile
import pandas as pd
import shutil
from zipfile import BadZipFile


def get_cookie(_url: str, _username: str, _password: str):
    _response = requests.post(url=_url, auth=HTTPBasicAuth(_username, _password), allow_redirects=False, stream=True)
    _cookies = requests.utils.dict_from_cookiejar(_response.cookies)
    return _cookies


def date_to_string(_date):
    return _date.strftime('%Y-%m-%d')

def cal_google_code(secret_key: str):
    secret_key += '=' * ((8 - len(secret_key) % 8) % 8)
    msg = struct.pack('>Q', int(time.time()) // 30)
    key = base64.b32decode(secret_key)
    hmac_digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = hmac_digest[-1] & 0x0F
    code = (struct.unpack('>I', hmac_digest[offset:offset+4])[0] & 0x7FFFFFFF) % 1000000
    return f"{code:06d}"

def save_csv(f_name, data):
    # 1. 创建文件对象
    _f = open(f_name, 'w', encoding='utf-8', newline='')
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(_f)
    # 4. 写入csv文件内容
    for row in data:
        csv_writer.writerow(row)
    # 5. 关闭文件
    _f.close()


def download_data(date, suffix, cookie, save_path,mode = 'binance-okx'):
    if mode == 'binance-okx':
        url = f'https://dcdl.digifinex.org/CF_data/ok_bn_cross_arbitrage/{date}/{date}04/{suffix}.csv'
    elif mode == 'binance-bybit':
        url = f'https://dcdl.digifinex.org/CF_data/bybit_bn_data/{date}/{date}04/{suffix}.csv'
    elif mode == 'binance-gate':
        url = f'https://dcdl.digifinex.org/CF_data/gate_bn_data/{date}/{date}04/{suffix}.csv'        
    print(f"Downloading {mode} data on {url} ")

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'{suffix}.csv')

    response = requests.get(url=url, cookies=cookie, stream=True)

    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"Download file to {save_path}")
    return response.status_code
import os
import platform

def get_base_dir():
    sys = platform.system().lower()
    if "windows" in sys:
        return "D:/"
    return "/Volumes/T7"



if __name__ == '__main__':

    path_mapping = {
        'binance-okx':'bn_ok',
        'binance-gate':'bn_gate',
        'binance-bybit':'bn_bybit'
    }
    key = 'YNW6YKQCA2YGAJVRZBREJ34BSI'
    password = cal_google_code(key)
    username = 'ray_xu'  # dcdl用户名
    cookie = get_cookie('https://dcdl.digifinex.org/', username, password)
    mode = 'binance-okx'
    date = '20260219'
    base_dir = get_base_dir()
    save_path = f'{base_dir}/Obentech/scored_df/{path_mapping[mode]}/{date}/'
    failed_files = set()

    suffix_list = [f'{date}04_scored_features_swap',f'{date}04_cfdc_dcpro1_limit_pos',f'{date}04_cfdc_dcpro5_limit_pos',f'{date}04_cfdc_dcpro11_limit_pos',f'{date}04_cfdc_dcpro17_limit_pos']
    suffix_list = [f'{date}04_scored_features_swap']
    # Download data
    print(f"\n=== Downloading {mode} data ===")
    for suffix in suffix_list:
        # Clear existing files for this symbol + env combination before downloading
        file_path = os.path.join(save_path, f'{suffix}.csv')
        try:
            status = download_data(date, suffix, cookie, save_path, mode)
            if status != 200:
                failed_files.add(f"{suffix}.csv")
        except Exception as e:
            print(f"Error downloading {suffix}: {e}")
            failed_files.add(f"{suffix}.csv")

    
    # Output summary
    if failed_files:
        print('\n[Summary] The following files failed to process:')
        for file in failed_files:
            print(file)
    else:
        print('\n[Summary] All files processed successfully.')