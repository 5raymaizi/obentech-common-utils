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


CROSS_EXCHANGE_MAP = {
    "okx-binance":"okx_binance",
    "bybit-binance":"bybit5_binance"
}

def download_data_history_depth(symbol, cookie, save_path, mode = 'ok-bn'):
    
    suffix = CROSS_EXCHANGE_MAP[mode]
    
    url = f'https://dcdl.digifinex.org/subscribe_to_csv/min_depth_hist/depth_{suffix}_{symbol.lower()}-usdt_1min.csv'
    print(f"Downloading history depth for {symbol} data")

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'depth_{suffix}_{symbol.lower()}-usdt_1min.csv')

    response = requests.get(url=url, cookies=cookie, stream=True)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    return response.status_code


if __name__ == '__main__':
    from CONFIG import *
    key = 'YNW6YKQCA2YGAJVRZBREJ34BSI'
    password = cal_google_code(key)
    username = 'ray_xu'  # dcdl用户名
    cookie = get_cookie('https://dcdl.digifinex.org/', username, password)

    symbol_list =['ATH','VVV']
    base_dir = get_base_dir()
    save_path = f'{base_dir}/Obentech/historyDepthData'
    failed_files = set()
    mode = 'bybit-binance'
    suffix = CROSS_EXCHANGE_MAP[mode]
    # Download data
    print("\n=== Downloading pnl analysis data ===")
    for symbol in symbol_list:
        # Clear existing files for this symbol + env combination before downloading
        file_path = os.path.join(save_path, f'depth_{suffix}_{symbol.lower()}-usdt_1min.csv')
        if os.path.exists(file_path):
            print(f"Clearing existing files for {symbol} in {file_path} for {mode}...")
            shutil.rmtree(file_path, ignore_errors=True) # 处理mac特有的._ETH_USDT.close.csv的情况

        try:
            status = download_data_history_depth(symbol, cookie, save_path, mode)
            if status != 200:
                failed_files.add(f"{symbol}.csv")
        except Exception as e:
            print(f"Error downloading {symbol}.csv: {e}") 
            failed_files.add(f"{symbol}.csv")

    
    # Output summary
    if failed_files:
        print('\n[Summary] The following files failed to process:')
        for file in failed_files:
            print(file)
    else:
        print('\n[Summary] All files processed successfully.')