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
from common_utils.CONFIG import * 

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


def download_data(symbol, env, suffix, cookie, save_path):


    url = f'https://dcdl.digifinex.org/CF_data/cf_dc/{env}/app/{symbol.upper()}USDT/{symbol.upper()}_USDT.{suffix}.csv'
    print(f"Downloading {env} {suffix} data for {symbol}: {url}")

    csv_dir = os.path.join(save_path, f'{symbol.upper()}USDT', env)
    os.makedirs(csv_dir, exist_ok=True)
    csv_full_path = os.path.join(csv_dir, f'{symbol.upper()}_USDT.{suffix}.csv')

    response = requests.get(url=url, cookies=cookie, stream=True)
    with open(csv_full_path, 'wb') as f:
        f.write(response.content)
    
    return response.status_code


if __name__ == '__main__':
    key = 'YNW6YKQCA2YGAJVRZBREJ34BSI'
    password = cal_google_code(key)
    username = 'ray_xu'  # dcdl用户名
    cookie = get_cookie('https://dcdl.digifinex.org/', username, password)
    base_dir = get_base_dir()

    symbol_list = ['ETH']
    env_list = ['manager_dcpro1','manager_dcpro2','manager_dcpro3','manager_dcpro4','manager_dcpro5','manager_dcpro7', 'manager_dcpro8','manager_dcpro9','manager_dcpro10','manager_dcpro11', 'manager_dcpro12','manager_dcpro13','manager_dcpro15','manager_dcpro16','manager_dcpro17','manager_dctest4','manager_maker_dc_pmpro_test','manager_dcpro24','manager_dcpro25','manager_dcpro26','manager_dcpro27','manager_dcpro29']
    env_list = ['manager_dcpro2','manager_dcpro3','manager_dcpro4','manager_dcpro5']
    # env_list = ['manager_maker_dc_pmpro_test','manager_dcpro1']
    suffix_list = ['open','close']
    # Save path configuration
    save_path = f'{base_dir}/Obentech/cfdcappData'
    failed_files = set()
    
    # Download OKX data
    print("\n=== Downloading app data ===")
    for symbol in symbol_list:
        for env in env_list:
            # Clear existing files for this symbol + env combination before downloading
            env_dir = os.path.join(save_path, f'{symbol.upper()}USDT', env)
            if os.path.exists(env_dir):
                print(f"Clearing existing files for {symbol} {env} in {env_dir}...")
                shutil.rmtree(env_dir, ignore_errors=True) # 处理mac特有的._ETH_USDT.close.csv的情况
            
            for suffix in suffix_list:
                try:
                    status = download_data(symbol, env, suffix, cookie, save_path)
                    if status != 200:
                        failed_files.add(f"{env}_{symbol}_{suffix}.csv")
                except Exception as e:
                    print(f"Error downloading {env} data for {symbol}: {e}")
                    failed_files.add(f"{env}_{symbol}_{suffix}.csv")

    
    # Output summary
    if failed_files:
        print('\n[Summary] The following files failed to process:')
        for file in failed_files:
            print(file)
    else:
        print('\n[Summary] All files processed successfully.')