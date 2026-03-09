#!/usr/bin/env python3
import csv
# from dateutil.relativedelta import relativedelta
import requests
from requests.auth import HTTPBasicAuth
import os
import hmac
import base64
import struct
import hashlib
import time
# from contextlib import closing
# from zipfile import ZipFile
# import pandas as pd
# from zipfile import BadZipFile


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


def download_funding_rate_data(symbol, exchange, cookie, save_path):
    if exchange == 'binance': 
        add1 = 'mark_price'
        add2 = f'{symbol.upper()}USDT.csv'
        add3 = 'exchange_market_data2'
    elif exchange == 'okx': 
        add1 = 'funding_rate'
        add2 = f'{symbol.upper()}-USDT-SWAP.csv'
        add3 = 'exchange_market_data2'
    elif exchange == 'bybit':
        add1 = 'funding_rate'
        add2 = f'{symbol.upper()}USDT.csv'
        add3 = 'exchange_market_data3'

    url = f'https://dcdl.digifinex.org/CF_data/{add3}/{exchange}/{add1}/{add2}'
    print(f"Downloading {exchange} data for {symbol}: {url}")

    csv_dir = os.path.join(save_path, exchange)
    os.makedirs(csv_dir, exist_ok=True)
    csv_full_path = os.path.join(csv_dir, add2)

    response = requests.get(url=url, cookies=cookie, stream=True)
    with open(csv_full_path, 'wb') as f:
        f.write(response.content)
    
    return response.status_code


if __name__ == '__main__':
    from common_utils.CONFIG import *
    key = 'YNW6YKQCA2YGAJVRZBREJ34BSI'
    password = cal_google_code(key)
    username = 'ray_xu'  # dcdl用户名
    cookie = get_cookie('https://dcdl.digifinex.org/', username, password)

    base_dir = get_base_dir()

    # ============ 配置区 ============
    mode = 'binance-okx'  # 'binance-okx' or 'binance-bybit'
    symbol_list = ['POWER']
    # ================================

    mode_to_exchanges = {
        'binance-okx': ['binance', 'okx'],
        'binance-bybit': ['binance', 'bybit'],
    }

    exchanges = mode_to_exchanges[mode]
    save_path = f'{base_dir}/Obentech/fundingRateData'
    failed_files = set()

    for exchange in exchanges:
        print(f"\n=== Downloading {exchange} data ===")
        for symbol in symbol_list:
            try:
                status = download_funding_rate_data(symbol, exchange, cookie, save_path)
                if status != 200:
                    failed_files.add(f"{exchange}_{symbol}")
            except Exception as e:
                print(f"Error downloading {exchange} data for {symbol}: {e}")
                failed_files.add(f"{exchange}_{symbol}")

    if failed_files:
        print('\n[Summary] The following files failed to process:')
        for file in failed_files:
            print(file)
    else:
        print('\n[Summary] All files processed successfully.')