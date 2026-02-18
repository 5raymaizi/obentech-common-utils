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
    # Original list for okx-binance pairs
    symbol_list = ['0G', '1INCH', '2Z', 'A', 'AAVE', 'ACE', 'ACH', 'ACT', 'ADA', 'AERO', 'AEVO', 'AGLD', 'AIXBT', 'ALGO', 'ALLO', 'ANIME', 'APE', 'API3', 'APR', 'APT', 'AR', 'ARB', 'ARKM', 'ASTER', 'AT', 'ATH', 'ATOM', 'AUCTION', 'AVAX', 'AVNT', 'AXS', 'BABY', 'BAND', 'BARD', 'BAT', 'BCH', 'BEAT', 'BERA', 'BICO', 'BIGTIME', 'BIO', 'BLUAI', 'BLUR', 'BNB', 'BOME', 'BRETT', 'BTC', 'CATI', 'CC', 'CELO', 'CETUS', 'CFX', 'CHZ', 'COAI', 'COMP', 'COOKIE', 'CRV', 'CVX', 'DASH', 'DEGEN', 'DOGE', 'DOOD', 'DOT', 'DYDX', 'EDEN', 'EGLD', 'EIGEN', 'ENA', 'ENJ', 'ENS', 'ENSO', 'ETC', 'ETH', 'ETHFI', 'ETHW', 'F', 'FARTCOIN', 'FIL', 'FLOW', 'FXS', 'GALA', 'GAS', 'GIGGLE', 'GLM', 'GMT', 'GMX', 'GPS', 'GRASS', 'GRT', 'H', 'HBAR', 'HMSTR', 'HOME', 'HUMA', 'HYPE', 'ICP', 'ICX', 'IMX', 'INIT', 'INJ', 'IOST', 'IOTA', 'IP', 'JCT', 'JELLYJELLY', 'JTO', 'JUP', 'KAITO', 'KGEN', 'KMNO', 'KSM', 'LA', 'LAB', 'LAYER', 'LDO', 'LIGHT', 'LINEA', 'LINK', 'LPT', 'LQTY', 'LRC', 'LTC', 'MAGIC', 'MANA', 'MASK', 'ME', 'MEME', 'MERL', 'MET', 'METIS', 'MEW', 'MINA', 'MMT', 'MON', 'MOODENG', 'MORPHO', 'MOVE', 'MUBARAK', 'NEAR', 'NEIRO', 'NEO', 'NIGHT', 'NMR', 'NOT', 'OL', 'OM', 'ONDO', 'ONE', 'ONT', 'OP', 'ORDER', 'ORDI', 'PARTI', 'PENDLE', 'PENGU', 'PEOPLE', 'PIEVERSE', 'PIPPIN', 'PLUME', 'PNUT', 'POL', 'POPCAT', 'PROMPT', 'PROVE', 'PUMP', 'PYTH', 'QTUM', 'RAVE', 'RECALL', 'RENDER', 'RESOLV', 'RLS', 'RSR', 'RVN', 'S', 'SAHARA', 'SAND', 'SAPIEN', 'SEI', 'SENT', 'SHELL', 'SIGN', 'SKY', 'SNX', 'SOL', 'SOLV', 'SOON', 'SOPH', 'SPK', 'SPX', 'SSV', 'STABLE', 'STRK', 'STX', 'SUI', 'SUSHI', 'SYRUP', 'TAO', 'THETA', 'TIA', 'TON', 'TRB', 'TREE', 'TRUMP', 'TRUST', 'TRUTH', 'TRX', 'TURBO', 'TURTLE', 'UMA', 'UNI', 'USELESS', 'VANA', 'VIRTUAL', 'W', 'WAL', 'WCT', 'WET', 'WIF', 'WLD', 'WLFI', 'WOO', 'XAN', 'XLM', 'XPL', 'XRP', 'XTZ', 'YB', 'YFI', 'YGG', 'ZBT', 'ZEC', 'ZEN', 'ZETA', 'ZIL', 'ZK', 'ZORA', 'ZRO', 'ZRX']

    
    # List for okx-bybit pairs
    symbols_okx_bybit = ['1INCH', 'A', 'AAVE', 'ACE', 'ACH', 'ACT', 'ADA', 'AERO', 'AEVO', 'AGLD', 'AI16Z', 'AIXBT', 'ALCH', 'ALGO', 'ANIME', 'APE', 'API3', 'APT', 'AR', 'ARB', 'ARC', 'ARKM', 'ATH', 'ATOM', 'AUCTION', 'AVAAI', 'AVAX', 'AXS', 'BABY', 'BADGER', 'BAL', 'BAND', 'BAT', 'BCH', 'BERA', 'BICO', 'BIGTIME', 'BIO', 'BLUR', 'BNB', 'BNT', 'BOME', 'BRETT', 'BTC', 'CATI', 'CELO', 'CETUS', 'CFX', 'CHZ', 'COMP', 'COOKIE', 'CORE', 'CRO', 'CRV', 'CTC', 'CVC', 'CVX', 'DEGEN', 'DGB', 'DOG', 'DOGE', 'DOGS', 'DOOD', 'DOT', 'DUCK', 'DYDX', 'EGLD', 'EIGEN', 'ENJ', 'ENS', 'ETC', 'ETH', 'ETHFI', 'ETHW', 'FARTCOIN', 'FIL', 'FLM', 'FLOW', 'FXS', 'GALA', 'GAS', 'GLM', 'GMT', 'GMX', 'GOAT', 'GODS', 'GPS', 'GRASS', 'GRIFFAIN', 'GRT', 'H', 'HBAR', 'HMSTR', 'HOME', 'HUMA', 'HYPE', 'ICP', 'ICX', 'ID', 'IMX', 'INIT', 'INJ', 'IOST', 'IOTA', 'IP', 'JELLYJELLY', 'JOE', 'JST', 'JTO', 'JUP', 'KAITO', 'KMNO', 'KSM', 'LA', 'LAUNCHCOIN', 'LDO', 'LINK', 'LOOKS', 'LPT', 'LQTY', 'LRC', 'LSK', 'LTC', 'MAGIC', 'MAJOR', 'MANA', 'MASK', 'ME', 'MEME', 'MERL', 'METIS', 'MEW', 'MINA', 'MKR', 'MOODENG', 'MORPHO', 'MOVE', 'MUBARAK', 'NEAR', 'NEIROETH', 'NEO', 'NIL', 'NMR', 'NOT', 'NXPC', 'OL', 'OM', 'ONDO', 'ONE', 'ONT', 'OP', 'ORBS', 'ORDI', 'PARTI', 'PENGU', 'PEOPLE', 'PERP', 'PIPPIN', 'PLUME', 'PNUT', 'POL', 'POPCAT', 'PRCL', 'PROMPT', 'PYTH', 'QTUM', 'RDNT', 'RENDER', 'RESOLV', 'RSR', 'RVN', 'S', 'SAHARA', 'SAND', 'SCR', 'SHELL', 'SIGN', 'SLP', 'SNX', 'SOL', 'SOLV', 'SONIC', 'SOON', 'SOPH', 'SPK', 'SSV', 'STORJ', 'STRK', 'STX', 'SUI', 'SUSHI', 'SWARMS','KGEN']
    symbols_bybit = ['1INCH', 'A', 'A2Z', 'AAVE', 'ACE', 'ACH', 'ACT', 'ACX', 'ADA', 'AERGO', 'AERO', 'AEVO', 'AGLD', 'AGT', 'AI', 'AI16Z', 'AIN', 'AIO', 'AIXBT', 'AKT', 'ALCH', 'ALGO', 'ALICE', 'ALPHA', 'ALPINE', 'ALT', 'ANIME', 'ANKR', 'APE', 'API3', 'APT', 'AR', 'ARB', 'ARC', 'ARK', 'ARKM', 'ARPA', 'ASR', 'ASTR', 'ATA', 'ATH', 'ATOM', 'AUCTION', 'AVA', 'AVAAI', 'AVAX', 'AWE', 'AXL', 'AXS', 'B', 'B2', 'B3', 'BABY', 'BAKE', 'BAN', 'BANANA', 'BANANAS31', 'BAND', 'BANK', 'BAT', 'BB', 'BCH', 'BDXN', 'BEL', 'BERA', 'BICO', 'BIGTIME', 'BIO', 'BLUR', 'BMT', 'BNB', 'BNT', 'BOME', 'BR', 'BRETT', 'BSV', 'BSW', 'BTC', 'BTR', 'C', 'C98', 'CAKE', 'CARV', 'CATI', 'CELO', 'CELR', 'CETUS', 'CFX', 'CGPT', 'CHESS', 'CHILLGUY', 'CHR', 'CHZ', 'CKB', 'COMP', 'COOKIE', 'COS', 'COTI', 'COW', 'CROSS', 'CRV', 'CTK', 'CTSI', 'CUDIS', 'CVC', 'CVX', 'CYBER', 'DAM', 'DASH', 'DEEP', 'DEGEN', 'DENT', 'DEXE', 'DIA', 'DMC', 'DOGE', 'DOGS', 'DOLO', 'DOOD', 'DOT', 'DRIFT', 'DUSK', 'DYDX', 'DYM', 'EDU', 'EGLD', 'EIGEN', 'ENA', 'ENJ', 'ENS', 'EPIC', 'EPT', 'ERA', 'ESPORTS', 'ETC', 'ETH', 'ETHFI', 'ETHW', 'F', 'FARTCOIN', 'FHE', 'FIDA', 'FIL', 'FIO', 'FIS', 'FLM', 'FLOW', 'FLUX', 'FORM', 'FORTH', 'FXS', 'G', 'GALA', 'GAS', 'GLM', 'GMT', 'GMX', 'GOAT', 'GPS', 'GRASS', 'GRIFFAIN', 'GRT', 'GTC', 'GUN', 'H', 'HAEDAL', 'HBAR', 'HEI', 'HFT', 'HIFI', 'HIGH', 'HIPPO', 'HIVE', 'HMSTR', 'HOME', 'HOOK', 'HOT', 'HUMA', 'HYPE', 'HYPER', 'ICNT', 'ICP', 'ICX', 'ID', 'ILV', 'IMX', 'IN', 'INIT', 'INJ', 'IO', 'IOST', 'IOTA', 'IOTX', 'IP', 'JASMY', 'JELLYJELLY', 'JOE', 'JST', 'JTO', 'JUP', 'KAIA', 'KAITO', 'KAS', 'KAVA', 'KDA', 'KERNEL', 'KMNO', 'KNC', 'KSM', 'LA', 'LDO', 'LINK', 'LISTA', 'LPT', 'LQTY', 'LRC', 'LSK', 'LTC', 'LUMIA', 'LUNA2', 'M', 'MAGIC', 'MANA', 'MANTA', 'MASK', 'MAV', 'MAVIA', 'MBOX', 'ME', 'MELANIA', 'MEME', 'MERL', 'METIS', 'MEW', 'MILK', 'MINA', 'MLN', 'MOCA', 'MOODENG', 'MORPHO', 'MOVE', 'MOVR', 'MTL', 'MUBARAK', 'MYRO', 'MYX', 'NAORIS', 'NEAR', 'NEIROETH', 'NEO', 'NEWT', 'NFP', 'NIL', 'NKN', 'NMR', 'NOT', 'NTRN', 'NXPC', 'OBOL', 'OG', 'OGN', 'OL', 'OM', 'OMNI', 'ONDO', 'ONE', 'ONG', 'ONT', 'OP', 'ORCA', 'ORDI', 'OXT', 'PARTI', 'PAXG', 'PENDLE', 'PENGU', 'PEOPLE', 'PERP', 'PHA', 'PHB', 'PIPPIN', 'PIXEL', 'PLUME', 'PNUT', 'POL', 'POLYX', 'PONKE', 'POPCAT', 'PORTAL', 'POWR', 'PROM', 'PROMPT', 'PROVE', 'PUFFER', 'PUMPBTC', 'PUNDIX', 'PYTH', 'QNT', 'QTUM', 'QUICK', 'RARE', 'RDNT', 'RED', 'RENDER', 'RESOLV', 'REZ', 'RIF', 'RLC', 'RONIN', 'ROSE', 'RPL', 'RSR', 'RUNE', 'RVN', 'S', 'SAFE', 'SAGA', 'SAHARA', 'SAND', 'SAPIEN', 'SCR', 'SCRT', 'SEI', 'SFP', 'SHELL', 'SIGN', 'SIREN', 'SKATE', 'SKL', 'SKYAI', 'SLERF', 'SLP', 'SNX', 'SOL', 'SOLV', 'SONIC', 'SOON', 'SOPH', 'SPELL', 'SPK', 'SPX', 'SQD', 'SSV', 'STEEM', 'STG', 'STO', 'STORJ', 'STRK', 'STX', 'SUI', 'SUN', 'SUPER', 'SUSHI', 'SWARMS', 'SWELL', 'SXP', 'SXT', 'SYN', 'SYRUP', 'SYS', 'T', 'TA', 'TAC', 'TAIKO', 'TANSSI', 'TAO', 'THE', 'THETA', 'TIA', 'TLM', 'TNSR', 'TOKEN', 'TON', 'TOWNS', 'TRB', 'TREE', 'TRU', 'TRUMP', 'TRX', 'TUT', 'TWT', 'UMA', 'UNI', 'USELESS', 'USUAL', 'UXLINK', 'VANA', 'VANRY', 'VELODROME', 'VELVET', 'VET', 'VIC', 'VINE', 'VIRTUAL', 'VOXEL', 'VTHO', 'VVV', 'W', 'WAL', 'WAXP', 'WCT', 'WIF', 'WLD', 'WOO', 'XAI', 'XCN', 'XLM', 'XMR', 'XNY', 'XRP', 'XTZ', 'XVG', 'XVS', 'YALA', 'YFI', 'YGG', 'ZEC', 'ZEN', 'ZEREBRO', 'ZETA', 'ZIL', 'ZK', 'ZORA', 'ZRC', 'ZRO', 'ZRX']
    # Convert lists to sets for efficient operations
    symbol_set = set(symbol_list)
    symbols_okx_bybit_set = set(symbols_okx_bybit)
    
    # Determine which symbols to download for each exchange
    binance_symbols = symbol_set  # All symbols in the original list for Binance
    okx_symbols = symbol_set.union(symbols_okx_bybit_set)  # All symbols for OKX (both lists)
    # binance_symbols = ['CHILLGUY','MEME','WIF','ATH','SYRUP','BB','KAIA','APE','IRYS','ALCH','STABLE','ADA','ASTER','SAPIEN']   
    # binance_symbols = ['CHILLGUY','CLO','STABLE']
    # binance_symbols = ['CYS','DASH','JELLYJELLY','JASMY','ASTR','XMR','MAGMA','AVNT','RENDER','BLUR','IMX','COMP','SYRUP','BB','AXS']
    # binance_symbols = ['LYN','ALCH','JELLYJELLY','CYS','STABLE','JASMY','GLM','AVNT','ZRO','BCH','GOAT','MEME','COMP','KMNO','CAKE','ASTR','EIGEN','KAIA'] 
    binance_symbols = ['ORDER', 'JTO', 'ID', 'TWT', 'BIGTIME', 'ASTR', 'ROSE', 'SYRUP', 'STG', 'BAT', 'POL','LYN','S','B2','AERO','SQD','FOGO']
    binance_symbols = ['IP','WLD','SPACE','WLFI']
    okx_symbols = binance_symbols
    bybit_symbols = symbols_bybit  # Only symbols in the okx-bybit list for Bybit
    # bybit_symbols = ['CHILLGUY','MEME','WIF','ATH','SYRUP','BB','KAIA','APE','IRYS','ALCH','STABLE','ADA','ASTER','SAPIEN']    
    bybit_symbols =  ['ORDER', 'JTO', 'ID', 'TWT', 'BIGTIME', 'ASTR', 'ROSE', 'SYRUP', 'STG', 'BAT', 'POL','LYN','S','B2','AERO','SQD','FOGO']
    bybit_symbols = ['IP','WLD','SPACE','WLFI']
    # Save path configuration
    save_path = f'{base_dir}/Obentech/fundingRateData'
    failed_files = set()
    
    # Download Binance data
    print("=== Downloading Binance data ===")
    for symbol in binance_symbols:
        try:
            status = download_funding_rate_data(symbol, 'binance', cookie, save_path)
            if status != 200:
                failed_files.add(f"binance_{symbol}")
        except Exception as e:
            print(f"Error downloading binance data for {symbol}: {e}")
            failed_files.add(f"binance_{symbol}")
    
    # # Download OKX data
    print("\n=== Downloading OKX data ===")
    for symbol in okx_symbols:
        try:
            status = download_funding_rate_data(symbol, 'okx', cookie, save_path)
            if status != 200:
                failed_files.add(f"okx_{symbol}")
        except Exception as e:
            print(f"Error downloading okx data for {symbol}: {e}")
            failed_files.add(f"okx_{symbol}")
    
    # Download Bybit data
    # print("\n=== Downloading Bybit data ===")
    # for symbol in bybit_symbols:
    #     try:
    #         status = download_funding_rate_data(symbol, 'bybit', cookie, save_path)
    #         if status != 200:
    #             failed_files.add(f"bybit_{symbol}")
    #     except Exception as e:
    #         print(f"Error downloading bybit data for {symbol}: {e}")
    #         failed_files.add(f"bybit_{symbol}")
    
    # Output summary
    if failed_files:
        print('\n[Summary] The following files failed to process:')
        for file in failed_files:
            print(file)
    else:
        print('\n[Summary] All files processed successfully.')