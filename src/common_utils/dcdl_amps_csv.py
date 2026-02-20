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
from typing import Union, List, Set, Optional

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


def download_data(env, ver, suffix, cookie, save_path):


    url = f'https://dcdl.digifinex.org/{env}/{ver}/order.{suffix}.csv'
    print(f"Downloading am {suffix} data: {url}")

    csv_dir = os.path.join(save_path, env, ver)
    os.makedirs(csv_dir, exist_ok=True)
    csv_full_path = os.path.join(csv_dir, f"order.{suffix}.csv")

    response = requests.get(url=url, cookies=cookie, stream=True)
    with open(csv_full_path, 'wb') as f:
        f.write(response.content)
    
    return response.status_code

def download_dcdl_orders(
    env: str,
    ver: str,
    suffix: Union[str, List[str]],
    *,
    save_path: str = "/Volumes/T7/Obentech/AMappData",
    username: str = "ray_xu",
    google_secret_key: str = "YNW6YKQCA2YGAJVRZBREJ34BSI",
    login_base_url: str = "https://dcdl.digifinex.org/",
    clear_existing: bool = True,
) -> Set[str]:
    """
    只需要传 env/ver/suffix（suffix 可为 str 或 list）
    返回：failed_files(set)，为空表示全部成功
    """

    suffix_list = [suffix] if isinstance(suffix, str) else list(suffix)

    # login -> cookie
    password = cal_google_code(google_secret_key)
    cookie = get_cookie(login_base_url, username, password)

    failed_files: Set[str] = set()

    for sfx in suffix_list:
        local_file = os.path.join(save_path, env, ver, f"order.{sfx}.csv")

        # 清理已存在的同名文件
        if clear_existing and os.path.exists(local_file):
            print(f"Clearing existing file: {local_file}")
            try:
                os.remove(local_file)
            except Exception as e:
                print(f"Failed to remove existing file: {local_file}, err={e}")

        try:
            status = download_data(env, ver, sfx, cookie, save_path)
            if status != 200:
                failed_files.add(f"{env}_{ver}_{sfx}.csv")
        except Exception as e:
            print(f"Error downloading {env}/{ver} order.{sfx}.csv: {e}")
            failed_files.add(f"{env}_{ver}_{sfx}.csv")

    # summary
    if failed_files:
        print("\n[Summary] The following files failed to process:")
        for f in sorted(failed_files):
            print(f)
    else:
        print("\n[Summary] All files processed successfully.")

    return failed_files


# if __name__ == '__main__':
#     key = 'YNW6YKQCA2YGAJVRZBREJ34BSI'
#     password = cal_google_code(key)
#     username = 'ray_xu'  # dcdl用户名
#     cookie = get_cookie('https://dcdl.digifinex.org/', username, password)

#     env = 'am_csv_s518'
#     ver = '20251227-am-UpdateAccountName-fixSaveEnabled'
#     suffix_list = ['btc_okx_binance_08_2','btc_okx_binance_09_2']
#     save_path = '/Volumes/T7/Obentech/AMappData'
#     failed_files = set()

#     for suffix in suffix_list:
#         # Clear existing files for this symbol + env combination before downloading
#         env_dir = os.path.join(save_path,env,ver, f"order.{suffix}.csv")
#         if os.path.exists(env_dir):
#             print(f"Clearing existing files for suffix...")
#             shutil.rmtree(env_dir, ignore_errors=True) # 处理mac特有的._ETH_USDT.close.csv的情况
        
#         for suffix in suffix_list:
#             try:
#                 status = download_data(env, ver, suffix, cookie, save_path)
#                 if status != 200:
#                     failed_files.add(f"{env}_{ver}_{suffix}.csv")
#             except Exception as e:
#                 print(f"Error downloading {env} data for version {ver} and symbol {suffix}: {e}")
#                 failed_files.add(f"{env}_{ver}_{suffix}.csv")

    
#     # Output summary
#     if failed_files:
#         print('\n[Summary] The following files failed to process:')
#         for file in failed_files:
#             print(file)
#     else:
#         print('\n[Summary] All files processed successfully.')




