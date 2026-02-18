#!/usr/bin/env python3
#
import ssl
from requests.adapters import HTTPAdapter  # 新增导入
from urllib3.poolmanager import PoolManager  # 新增导入
#
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

def cal_google_code(secret_key: str):
    # secret key 的长度必须是 8 的倍数。所以如果 secret key 不符合要求，需要在后面补上相应个数的 "="
    secret_key_len = len(secret_key)
    secret_key_pad_len = math.ceil(secret_key_len / 8) * 8 - secret_key_len
    secret_key = secret_key + "=" * secret_key_pad_len

    duration_input = int(time.time()) // 30
    _key = base64.b32decode(secret_key)
    msg = struct.pack(">Q", duration_input)
    google_code = hmac.new(_key, msg, hashlib.sha1).digest()
    o = google_code[19] & 15
    google_code = str((struct.unpack(">I", google_code[o:o+4])[0] & 0x7fffffff) % 1000000)

    # 生成的验证码未必是 6 位，注意要在前面补 0
    if len(google_code) == 5:  # Only if length of the code is 5, a zero will be added at the beginning of the code.
        google_code = '0' + google_code
    return google_code

def get_cookie(_url: str, _username: str, _password: str):
    _response = requests.post(url=_url, auth=HTTPBasicAuth(_username, _password), allow_redirects=False, stream=True)
    _cookies = requests.utils.dict_from_cookiejar(_response.cookies)
    return _cookies


def date_to_string(_date):
    return _date.strftime('%Y-%m-%d')


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


##############
    # ========== 新增 TLSAdapter 类 ==========
class TLSAdapter(HTTPAdapter):
    """强制使用 TLS 1.2+ 的自定义适配器"""
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.options |= ssl.OP_NO_SSLv3  # 禁用SSLv3
        context.options |= ssl.OP_NO_TLSv1   # 禁用TLS1.0
        context.options |= ssl.OP_NO_TLSv1_1 # 禁用TLS1.1
        context.set_ciphers('DEFAULT@SECLEVEL=2')
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# ========== 修改 get_cookie 函数 ==========
def get_cookie(_url: str, _username: str, _password: str):
    # 创建自定义Session并挂载适配器
    session = requests.Session()
    session.mount("https://", TLSAdapter())  # 挂载到所有HTTPS请求
    _response = session.post(
        url=_url,
        auth=HTTPBasicAuth(_username, _password),
        allow_redirects=False,
        stream=True
    )
    return session  # 返回整个Session（包含Cookie和TLS配置）
##############
if __name__ == '__main__':
    key = 'XXX' # dcdl密钥
    # password = cal_google_code(key)
    password = '765696'
    username = 'ray_xu'  # dcdl用户名
    session = get_cookie('https://dcdl.digifinex.org/', username, password)
    # proxies = {'https': 'https://127.0.0.0.0:7890', 'http': 'http://127.0.0.1:7890'}
    print("Cookies:", session.cookies.get_dict())  # 打印验证Cookie
    start_date = datetime.date(2025, 4, 21)
    end_date = datetime.date(2025, 4, 22)
    path = f'/Users/rayxu/Desktop/Obentech/temp'
    symbol_list = ['BTC','ETH','DOGE','XRP','SOL','DOGE','TON','ONDO']
    symbol_list = ['BNB', 'TRX', 'ADA', 'LEO', 'AVAX', 'LINK', 'XLM', 'SUI', 'HBAR', 'SHIB', 'BCH', 'LTC', 'DOT', 'HYPE', 'BGB', 'PI']
    symbol_list = ['SOL']
    symbol_list = ['BTC','ETH','XRP','SOL','DOGE','TON','ONDO','AAVE','CRO','ICP','LEO','MNT','OKB','PI','SHIB','TKX','TRX','WBT','XLM','XMR']
    # symbol_list = ['WBT', 'XMR', 'UNI', 'CBBTC', 'PEPE', 'OKB', 'APT', 'GT', 'TKX',
    # 'NEAR', 'ICP', 'CRO', 'MNT', 'ETC', 'AAVE']
    data_types = ['depth']
    trade_types = ['spot','swap']
    exchanges = ['binance','bybit5','okx']
    exchanges = ['binance','bybit5']
    save_path = f'/Users/rayxu/Desktop/Obentech/dcdlData'
    current_date = start_date

    failed_files = set()

    while current_date <= end_date:
        for symbol in symbol_list:
            for data_type in data_types:
                for trade_type in trade_types:
                    for exchange in exchanges:
                        if data_type == 'trade':
                            add = ''
                            add1 = ''
                        else:
                            add = '_5_100'
                            add1 = '5'
                        url = f'https://dcdl.digifinex.org/subscribe_to_csv/{date_to_string(current_date)}/{date_to_string(current_date)}_{data_type}_{exchange}_{trade_type}_{symbol.lower()}-usdt{add}.zip'
                        print(url)

                        path1 = f'{path}/{date_to_string(current_date)}_{data_type}_{exchange}_{trade_type}_{symbol.lower()}-usdt{add}.zip'
                        if os.path.exists(path1):
                            print(f'[Skip] File already exists: {path1}')
                        else:
                            response = session.get(url=url, stream=True)  # 使用配置好的Session
                            with open(path1, 'wb') as f:
                                f.write(response.content)

                        zip_file = f'{date_to_string(current_date)}_{data_type}_{exchange}_{trade_type}_{symbol.lower()}-usdt{add}.zip'
                        csv_file = f'{symbol.lower()}usdt_{date_to_string(current_date)}_{data_type}{add1}.csv'
                        csv_file1 = f'{data_type}_{exchange}_{trade_type}_{symbol.lower()}-usdt{add}.csv'

                        try:
                            with ZipFile(path1) as zf:
                                with zf.open(csv_file1) as file:
                                    df = pd.read_csv(file)

                                    if data_type == 'trade': 
                                        save_path = f'/Users/rayxu/Desktop/Obentech/dcdlData/{exchange}/trade/{symbol}/{trade_type}/'
                                        os.makedirs(save_path, exist_ok=True)
                                        df['e'] = 'trade'
                                        df['E'] = df['event time']
                                        df['type'] = df['trade type']
                                        df['s'] = f'{symbol.upper()}USDT'
                                        df.loc[df['side'] == 'sell', 'm'] = True
                                        df.loc[df['side'] == 'buy', 'm'] = False
                                        df['local time'] = df['local time'].apply(lambda x: '2025-' + x)
                                        df = df[['local time', 'e', 'E', 'id', 's', 'seller id', 'price', 'amount', 'type', 'm']]
                                        df.columns = ['ts','e','E','T','s','a','p','q','type','m']
                                        df.to_csv(save_path + csv_file, index=0)

                                    elif data_type == 'depth':
                                        save_path = f'/Users/rayxu/Desktop/Obentech/dcdlData/{exchange}/books/{symbol}/{trade_type}/'
                                        os.makedirs(save_path, exist_ok=True)
                                        df['E'] = df['event time']
                                        df['local time'] = df['local time'].apply(lambda x: '2025-' + x)
                                        df = df[['local time','E',  'exchange time', 'bid price 1', 'bid amount 1',
                                                'bid price 2', 'bid amount 2', 'bid price 3', 'bid amount 3',
                                                'bid price 4', 'bid amount 4', 'bid price 5', 'bid amount 5',
                                                'ask price 1', 'ask amount 1', 'ask price 2', 'ask amount 2',
                                                'ask price 3', 'ask amount 3', 'ask price 4', 'ask amount 4',
                                                'ask price 5', 'ask amount 5']]
                                        df.columns = ['received_time','E','T','bid_price0','bid_size0','bid_price1','bid_size1','bid_price2','bid_size2',
                                                    'bid_price3','bid_size3','bid_price4','bid_size4','ask_price0','ask_size0','ask_price1','ask_size1','ask_price2','ask_size2','ask_price3','ask_size3','ask_price4','ask_size4']
                                        df.to_csv(save_path + csv_file, index=0)
                        except (BadZipFile, KeyError, pd.errors.ParserError) as e:
                            print(f'[Error] {zip_file} failed: {e}')
                            trimmed = '_'.join(zip_file.split('_')[1:])
                            failed_files.add(trimmed)
        current_date += relativedelta(days=1)

    # 输出失败文件
    if failed_files:
        print('\n[Summary] The following files failed to process:')
        for file in failed_files:
            print(file)
    else:
        print('\n[Summary] All files processed successfully.')