import pytz
import os
import platform
import pandas as pd
from datetime import datetime, timedelta
# test02061823pm

# ====== 全局配置 ======

if 's517' in platform.node():
    print("Running on Server")
    IS_SERVER = True
else:
    IS_SERVER = False
def get_base_dir():
    sys = platform.system().lower()
    if "windows" in sys:
        return "D:"
    return "/Volumes/T7"


base_dir = get_base_dir()

if IS_SERVER:

    DATA_BASE_DIR = "/data/vhosts/future_analysis/data"
    REPORT_BASE_DIR = "/data/vhosts/future_analysis/reports"
    LOG_BASE_DIR = "/data/vhosts/future_analysis/logs"


    DATA_BASE_DIR_BN_BYBIT = "/data/vhosts/future_analysis_bybit_bn/data"
    REPORT_BASE_DIR_BN_BYBIT = "/data/vhosts/future_analysis_bybit_bn/reports"
    LOG_BASE_DIR_BN_BYBIT = "/data/vhosts/future_analysis_bybit_bn/logs"
    
    DATA_BASE_DIR_BN_GATE = "/data/vhosts/future_analysis_gate_bn/data"
    REPORT_BASE_DIR_BN_GATE = "/data/vhosts/future_analysis_gate_bn/reports"
    LOG_BASE_DIR_BN_GATE = "/data/vhosts/future_analysis_gate_bn/logs"
  
    PNL_ANALYSIS_DIR = "/data_file/CF_data/pnl_analysis"

    FUNDING_RATE_OKX_DIR     = f"/data_file/exchange_market_data/okx/funding_rate/"
    FUNDING_RATE_BINANCE_DIR = f"/data_file/exchange_market_data/binance/mark_price/"
    FUNDING_RATE_BYBIT_DIR = f"/data_file/exchange_market_data3/bybit/funding_rate/"
    FUNDING_RATE_GATE_DIR = f"/data_file/exchange_market_data3/gateio/funding_rate/"

    MINUTE_DEPTH_DIR = '/data_file/subscribe_to_csv/min_depth_hist/'    

    os.makedirs(REPORT_BASE_DIR, exist_ok=True)
    os.makedirs(LOG_BASE_DIR, exist_ok=True)

else:

    DATA_BASE_DIR = "/Users/rayxu/Downloads"
    DATA_BASE_DIR = f"{base_dir}/Obentech/scored_df/bn_ok/"
    DATA_BASE_DIR_BN_BYBIT = f"{base_dir}/Obentech/scored_df/bn_bybit/"
    DATA_BASE_DIR_BN_GATE = f"{base_dir}/Obentech/scored_df/bn_gate/"


    REPORT_BASE_DIR = f"{base_dir}/Obentech/future_analysis/reports"
    LOG_BASE_DIR = f"{base_dir}/Obentech/future_analysis/logs"
    PNL_ANALYSIS_DIR = f"{base_dir}/Obentech/pnl_analysis"



    FUNDING_RATE_OKX_DIR = f"{base_dir}/Obentech/fundingRateData/okx/"
    FUNDING_RATE_BINANCE_DIR = f"{base_dir}/Obentech/fundingRateData/binance/"
    FUNDING_RATE_BYBIT_DIR = f"{base_dir}/Obentech/fundingRateData/bybit/"
    FUNDING_RATE_GATE_DIR = f"{base_dir}/Obentech/fundingRateData/gate/"

    MINUTE_DEPTH_DIR = f"{base_dir}/Obentech/historyDepthData/"

    os.makedirs(REPORT_BASE_DIR, exist_ok=True)
    os.makedirs(LOG_BASE_DIR, exist_ok=True)

tz_bj = pytz.timezone("Asia/Shanghai") # 默认使用北京时间
now_bj = datetime.now(tz_bj) # eg: datetime.datetime(2025, 12, 19, 8, 7, 23, 854260, tzinfo=<DstTzInfo 'Asia/Shanghai' CST+8:00:00 STD>)
today_bj = now_bj.date() # eg: datetime.date(2025, 12, 19)

dcdl_date_str = today_bj.strftime("%Y%m%d")  # '20251219'
data_dir = os.path.join(DATA_BASE_DIR,dcdl_date_str,f"{dcdl_date_str}04") if IS_SERVER else DATA_BASE_DIR  # eg: '/data/vhosts/future_analysis/data/20251219/2025121904' on SERVER; '/Users/rayxu/Downloads' on LOCAL


et_bj = tz_bj.localize(datetime(today_bj.year, today_bj.month, today_bj.day, 4, 5, 0))   # 当天北京时间 04:05:00

st_bj = et_bj - timedelta(days=3) 

st_str = st_bj.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
et_str = et_bj.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
last_time_cf = pd.to_datetime(et_bj.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"))

# Lark 通知配置

# LARK_BOT_ID = 'f09b9fdf-6382-435a-a148-7a56a75ceaba'
APP_ID = "cli_a9b34d9ecd389ed1"        
APP_SECRET = "WitgMhaM39DB4NhVCvhzMbRHjF1iHhlr"  
LARK_CHAT_ID = "oc_a1d3ef06f832c541636aad63f9ecf9d1"  




# ====== Symbol Lists (按交易所对分类) ======

# BN-OKX 支持的币种列表（最全）
SYMBOL_LIST_BN_OKX = '0G,1INCH,2Z,A,AAVE,ACE,ACH,ACT,ADA,AERO,AEVO,AGLD,AIXBT,ALGO,ALLO,ANIME,APE,API3,APR,APT,AR,ARB,ARKM,ASTER,AT,ATH,ATOM,AUCTION,AVAX,AVNT,AXS,BABY,BAND,BARD,BAT,BCH,BEAT,BERA,BICO,BIGTIME,BIO,BLUAI,BLUR,BNB,BOME,BRETT,BREV,BTC,CATI,CC,CELO,CFX,CHZ,COAI,COMP,COOKIE,CRV,CVX,DASH,DOGE,DOOD,DOT,DYDX,EDEN,EGLD,EIGEN,ENA,ENJ,ENS,ENSO,ETC,ETH,ETHFI,ETHW,F,FARTCOIN,FIL,FLOW,FOGO,GALA,GAS,GIGGLE,GLM,GMT,GMX,GPS,GRASS,GRT,H,HBAR,HMSTR,HOME,HUMA,HYPE,ICP,ICX,IMX,INIT,INJ,IOST,IOTA,IP,JCT,JELLYJELLY,JTO,JUP,KAITO,KGEN,KMNO,KSM,LA,LAB,LAYER,LDO,LIGHT,LINEA,LINK,LIT,LPT,LQTY,LRC,LTC,MAGIC,MANA,MASK,ME,MEME,MERL,MET,METIS,MEW,MINA,MMT,MON,MOODENG,MORPHO,MOVE,MUBARAK,NEAR,NEIRO,NEO,NIGHT,NMR,NOT,OL,OM,ONDO,ONE,ONT,OP,ORDER,ORDI,PARTI,PENDLE,PENGU,PEOPLE,PIEVERSE,PIPPIN,PLUME,PNUT,POL,POPCAT,PROMPT,PROVE,PUMP,PYTH,QTUM,RAVE,RECALL,RENDER,RESOLV,RIVER,RLS,RSR,RVN,S,SAHARA,SAND,SAPIEN,SEI,SENT,SHELL,SIGN,SKY,SNX,SOL,SOLV,SOON,SOPH,SPK,SPX,SSV,STABLE,STRK,STX,SUI,SUSHI,SYRUP,TAO,THETA,TIA,TON,TRB,TREE,TRUMP,TRUST,TRUTH,TRX,TURBO,TURTLE,UMA,UNI,USELESS,VANA,VIRTUAL,W,WAL,WCT,WET,WIF,WLD,WLFI,WOO,XAN,XLM,XPL,XRP,XTZ,YB,YFI,YGG,ZAMA,ZBT,ZEC,ZEN,ZETA,ZIL,ZK,ZKP,ZORA,ZRO,ZRX'.split(',')
# BN-BYBIT 支持的币种列表（Bybit 支持的币种可能略少）
SYMBOL_LIST_BN_BYBIT = '0G,1INCH,2Z,4,A,A2Z,AAVE,ACE,ACH,ACT,ACX,ADA,AERGO,AERO,AEVO,AGLD,AIN,AIO,AIXBT,AKE,AKT,ALCH,ALGO,ALICE,ALLO,ALPINE,ALT,ANIME,ANKR,APE,API3,APR,APT,AR,ARB,ARC,ARIA,ARK,ARKM,ARPA,ASR,ASTER,ASTR,AT,ATH,ATOM,AUCTION,AVA,AVAAI,AVAX,AVNT,AWE,AXL,AXS,B,B2,B3,BABY,BAN,BANANA,BANANAS31,BAND,BANK,BARD,BAT,BB,BCH,BEAT,BEL,BERA,BICO,BIGTIME,BIO,BLESS,BLUAI,BLUR,BMT,BNB,BNT,BOME,BR,BRETT,BREV,BSV,BTC,BTR,C,C98,CAKE,CARV,CATI,CC,CELO,CETUS,CFX,CGPT,CHILLGUY,CHR,CHZ,CKB,CLANKER,CLO,COAI,COMMON,COMP,COOKIE,COTI,COW,CROSS,CRV,CTK,CTSI,CUDIS,CVC,CVX,CYBER,CYS,DASH,DEEP,DEGEN,DENT,DEXE,DIA,DOGE,DOLO,DOOD,DOT,DRIFT,DUSK,DYDX,DYM,EDEN,EDU,EGLD,EIGEN,ENA,ENJ,ENS,ENSO,EPIC,EPT,ERA,ESPORTS,ETC,ETH,ETHFI,EUL,EVAA,F,FARTCOIN,FF,FHE,FIDA,FIL,FIO,FLOCK,FLOW,FLUID,FLUX,FOLKS,FORM,GALA,GAS,GIGGLE,GLM,GMT,GMX,GOAT,GPS,GRASS,GRIFFAIN,GRT,GUN,H,HAEDAL,HANA,HBAR,HEI,HEMI,HFT,HIGH,HIPPO,HIVE,HMSTR,HOLO,HOME,HOOK,HUMA,HYPE,HYPER,ICNT,ICP,ICX,ID,ILV,IMX,IN,INIT,INJ,IO,IOST,IOTA,IOTX,IP,IRYS,JASMY,JCT,JELLYJELLY,JST,JTO,JUP,KAIA,KAITO,KAS,KAVA,KERNEL,KGEN,KITE,KMNO,KNC,KSM,LA,LAB,LDO,LIGHT,LINEA,LINK,LISTA,LIT,LPT,LQTY,LRC,LSK,LTC,LUMIA,LUNA2,LYN,M,MAGIC,MAGMA,MANA,MANTA,MASK,MAV,MAVIA,MBOX,ME,MELANIA,MEME,MERL,MET,METIS,MEW,MINA,MIRA,MITO,MLN,MMT,MOCA,MON,MOODENG,MORPHO,MOVE,MOVR,MTL,MUBARAK,MYX,NAORIS,NEAR,NEO,NEWT,NFP,NIGHT,NIL,NKN,NMR,NOM,NOT,NTRN,NXPC,OG,OGN,OL,OM,ONDO,ONE,ONG,ONT,OP,OPEN,ORCA,ORDER,ORDI,OXT,PARTI,PAXG,PENDLE,PENGU,PEOPLE,PHA,PHB,PIEVERSE,PIPPIN,PIXEL,PLUME,PNUT,POL,POLYX,POPCAT,PORTAL,POWER,POWR,PROM,PROMPT,PROVE,PTB,PUFFER,PUMPBTC,PUNDIX,PYTH,Q,QNT,QTUM,RARE,RAVE,RDNT,RECALL,RED,RENDER,RESOLV,REZ,RIVER,RLC,RLS,RONIN,ROSE,RPL,RSR,RUNE,RVN,S,SAFE,SAGA,SAHARA,SAND,SAPIEN,SCR,SCRT,SEI,SFP,SHELL,SIGN,SIREN,SKL,SKY,SKYAI,SLP,SNX,SOL,SOLV,SOMI,SONIC,SOON,SOPH,SPELL,SPK,SPX,SQD,SSV,STABLE,STBL,STEEM,STG,STO,STORJ,STRK,STX,SUI,SUN,SUPER,SUSHI,SWARMS,SXT,SYN,SYRUP,SYS,T,TA,TAC,TAIKO,TAO,THE,THETA,TIA,TLM,TNSR,TON,TOWNS,TRB,TREE,TRU,TRUMP,TRUST,TRUTH,TRX,TURTLE,TUT,TWT,UAI,UB,UMA,UNI,US,USELESS,USUAL,VANA,VANRY,VELODROME,VELVET,VET,VFY,VIC,VINE,VIRTUAL,VVV,W,WAL,WAXP,WCT,WET,WIF,WLD,WLFI,WOO,XAI,XAN,XLM,XMR,XNY,XPIN,XPL,XRP,XTZ,XVG,XVS,YALA,YB,YFI,YGG,ZBT,ZEC,ZEN,ZEREBRO,ZETA,ZIL,ZK,ZKC,ZKP,ZORA,ZRO,ZRX'.split(',')
# BN-GATE 支持的币种列表（Gate 支持的币种）
SYMBOL_LIST_BN_GATE = 'JELLYJELLY,COAI,RECALL,AT,GIGGLE,ALGO,STRK,USELESS,S,BLUAI,BABY,JOE,ETHW,TURBO,POL,2Z,AR,IOST,APR,OL,GALA,BOME,MEW,F,KMNO,AUCTION,BLUR,MOODENG,SPX,PROVE,BAND,INJ,KAITO,AIXBT,WOO,BCH,CHZ,AI16Z,GMT,BAT,PYTH,WAL,COOKIE,ACT,SONIC,YGG,AERO,LRC,NMR,JTO,TAO,OM,APT,SYRUP,STX,TRX,BNT,ZETA,MET,BICO,TREE,TON,CATI,AXS,KITE,METIS,BRETT,ZIL,FARTCOIN,FIL,ENSO,H,HYPE,BIO,KGEN,MERL,THETA,TIA,API3,ADA,WLFI,BIGTIME,EGLD,HBAR,RENDER,ZK,CFX,LTC,LPT,XPL,APE,ME,XAN,1INCH,VIRTUAL,DEGEN,ZRO,GRT,MORPHO,ENS,PENGU,SAHARA,IOTA,PNUT,ETH,NEO,HOME,TURTLE,ETHFI,XLM,JUP,FXS,TRB,INIT,PUMP,CETUS,ONDO,SUI,AAVE,TRUMP,EIGEN,NEWT,IP,HUMA,VANA,BNB,ZORA,LDO,SHELL,BTC,IMX,XRP,MOVE,LA,LQTY,ACE,ATOM,AVAX,YB,OP,LINK,NEAR,ETC,AVNT,SOLV,WIF,SOL,RSR,SSV,WLD,SAND,DOGE,GOAT,ARKM,SIGN,BERA,UNI,PLUME,SPK,W,AEVO,GAS,GLM,ASTER,SKY,MUBARAK,BARD,SOPH,DOT,CELO,YFI,ENA,RESOLV,NEIRO,KSM,SUSHI,ORDI,SNX,GMX,NOT,AGLD,PROMPT,EDEN,SOON,GRASS,COMP,MANA,ICX,MAGIC,LAYER,MEME,POPCAT,DYDX,ZRX,A,ANIME,PEOPLE,DOOD,ICP,ONE,ARB,UMA,PARTI,LINEA,GPS,CRV,WCT,0G,FLOW,MINA,RVN,QTUM,ZBT,PENDLE,CVC,CVX,ACH,ORDER,MASK,ATH,XTZ'.split(',')


BLACK_LIST = [
    'ZEREBRO', 'WAXP', 'NEIROETH', 'ALCH', 'GMT', 'GRT', 'PENGU', 'HUMA', 'ORDI',
    'DEGEN', 'MKR', 'SSV', 'BLUR', 'CTC', 'LPT', 'CHZ', 'XAUT', 'PROMPT', 'CSPR',
    'TRB', 'RENDER', 'AEVO', 'ARB', 'ACT', 'KMNO', 'ATOM', 'CETUS', 'WIF', 'BIGTIME',
    'SNX', 'BABY', 'AUCTION', 'RAY', 'BONK', 'NEO', 'LRC', 'ZK', 'COMP', 'AAVE',
    'THETA', 'LTC', 'OP', 'ENS', 'TRX', 'AR', 'APT', 'CFX', 'XRP', 'BTC', 'UNI',
    'ARKM', 'SHIB', 'ADA', 'DOT', 'EIGEN', 'ETH', 'DOGE', 'SOL']

BLACK_LIST = []


# ====== 交易所对配置 ======
# 添加新的交易所对时，只需在这里添加配置，程序无需修改
EXCHANGE_PAIR_CONFIG = {
    'binance-okx': {
        'data_base_dir': DATA_BASE_DIR if IS_SERVER else DATA_BASE_DIR,
        'report_base_dir': REPORT_BASE_DIR,
        'log_base_dir': LOG_BASE_DIR,
        'funding_rate_dirs': {
            'exchange1': FUNDING_RATE_BINANCE_DIR,  # 第一个交易所
            'exchange2': FUNDING_RATE_OKX_DIR       # 第二个交易所
        },
        'mode': 'BN-OKX',  # 用于 analyze_funding_rate_diff_v2
        'env_list': ['dcpro1', 'dcpro5', 'dcpro17', 'dcpro11','dcpro2','dcpro3'],  # limit_pos 文件的环境列表
        'env_thresholds': {
            'dcpro1': 200,   # dcpro1 的阈值
            'dcpro5': 1000,  # 其他环境的阈值
            'dcpro17': 1000,
            'dcpro11': 1000,
            'dcpro2':5000,
            'dcpro3':5000
        },
        'pnl_env': 'dcpro1',  # PNL 报告使用的环境（对应 PORTFOLIO_CONFIG 中的键）
        'pnl_dict_file': 'symbol_pnl_dict_100WU.pkl',
        'symbol_list_funding': SYMBOL_LIST_BN_OKX,  # 资金费率分析的币种列表
        'display_name': 'BN-OK'
    },
    'binance-bybit': {
        'data_base_dir': DATA_BASE_DIR_BN_BYBIT if IS_SERVER else DATA_BASE_DIR_BN_BYBIT,
        'report_base_dir': REPORT_BASE_DIR_BN_BYBIT if IS_SERVER else REPORT_BASE_DIR,
        'log_base_dir': LOG_BASE_DIR_BN_BYBIT if IS_SERVER else LOG_BASE_DIR,
        'funding_rate_dirs': {
            'exchange1': FUNDING_RATE_BINANCE_DIR,
            'exchange2': FUNDING_RATE_BYBIT_DIR
        },
        'mode': 'BN-BYBIT',
        'env_list': ['dcbb1'],  # bybit 只有一个环境
        'env_thresholds': {
            'pmtest2': 200,
            'dcbb1':5000,
        },
        'pnl_env': 'dcbb1',
        'pnl_dict_file': 'symbol_pnl_dict_dcbb1.pkl',
        'symbol_list_funding': SYMBOL_LIST_BN_BYBIT,  # Bybit 支持的币种列表
        'display_name': 'Bybit-BN'
    },
    'binance-gate': {
        'data_base_dir': DATA_BASE_DIR_BN_GATE if IS_SERVER else DATA_BASE_DIR_BN_GATE,
        'report_base_dir': REPORT_BASE_DIR_BN_GATE if IS_SERVER else REPORT_BASE_DIR,
        'log_base_dir': LOG_BASE_DIR_BN_GATE if IS_SERVER else LOG_BASE_DIR,
        'funding_rate_dirs': {
            'exchange1': FUNDING_RATE_BINANCE_DIR,
            'exchange2': FUNDING_RATE_GATE_DIR
        },
        'mode': 'BN-GATE',
        'env_list': ['pmtest4'],  # gate 使用 pmtest4 环境
        'env_thresholds': {
            'pmtest4': 200
        },
        'pnl_env': 'pmtest4',
        'pnl_dict_file': 'symbol_pnl_dict_pmtest4.pkl',
        'symbol_list_funding': SYMBOL_LIST_BN_GATE,  # Gate 支持的币种列表
        'display_name': 'Gate-BN'
    }
}

# 默认使用的交易所对
DEFAULT_EXCHANGE_PAIR = 'binance-okx'




# 各个环境的本金

PORTFOLIO_CONFIG = {
    'pmpro': {
        'base_path': '/data/vhosts/cf_dc/manager_maker_dc_pmpro_test/app',
        'total_capital': 100000,
        'title_prefix': '10W U Portfolio PnL',
        'file_suffix': '_10WU',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-05-22 06:00:00'
    },
    'dcpro1': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro1/app',
        'total_capital': 1000000,
        'title_prefix': 'Pro1(100WU) PnL',
        'file_suffix': '_100WU',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-06-25 04:00:00'
    },
    'dcpro2': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro2/app',
        'total_capital': 2100000,
        'title_prefix': 'Pro2 (210WU) PnL',
        'file_suffix': '_dcpro2',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-07-29 22:00:00'
    },
    # 'dcob1': {
    #     'base_path': '/data_file/CF_data/cf_dc/manager_dcob1/app',
    #     'total_capital': 1000000,
    #     'title_prefix': '100W U OK-Bybit Portfolio PnL',
    #     'file_suffix': '_dcob1',
    #     'denominator_ratio': 0.02,
    #     'long_term_start': '2025-07-22 19:00:00'
    # },
    'dcpro3': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro3/app',
        'total_capital': 3800000,
        'title_prefix': 'Pro3(380WU) PnL',
        'file_suffix': '_dcpro3',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-07-17 19:00:00'
    },
    'dcpro4': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro4/app',
        'total_capital': 3100000,
        'title_prefix': 'Pro4(310WU) PnL',
        'file_suffix': '_dcpro4',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-07-29 22:00:00'
    },
    'dcpro5': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro5/app',
        'total_capital': 410000,
        'title_prefix': 'Pro5(41WU) PnL',
        'file_suffix': '_dcpro5',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-08-02 00:00:00'
    },
    'dcpro6': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro6/app',
        'total_capital': 2000000,
        'title_prefix': 'Pro6(23BTC) Portfolio PnL',
        'file_suffix': '_dcpro6',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-08-16 00:00:00'
    },
    'dcpro7': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro7/app',
        'total_capital': 2300000,
        'title_prefix': 'Pro7(230WU) PnL',
        'file_suffix': '_dcpro7',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-08-16 00:00:00'
    },
    'dcpro8': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro8/app',
        'total_capital': 2000000,
        'title_prefix': 'Pro8(200WU) PnL',
        'file_suffix': '_dcpro8',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-09-12 04:00:00'
    },
    'dcpro9': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro9/app',
        'total_capital': 2000000,
        'title_prefix': 'Pro9(200WU) PnL',
        'file_suffix': '_dcpro9',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-09-12 04:00:00'
    },
    'dcpro10': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro10/app',
        'total_capital': 3000000,
        'title_prefix': 'Pro10(300WU) PnL',
        'file_suffix': '_dcpro10',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-09-12 04:00:00'
    },
    'dcpro11': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro11/app',
        'total_capital': 900000,
        'title_prefix': 'Pro11(90WU) PnL',
        'file_suffix': '_dcpro11',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-09-13 16:00:00'
    },
    'dcpro12': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro12/app',
        'total_capital': 1400000,
        'title_prefix': 'Pro12(140WU) PnL',
        'file_suffix': '_dcpro12',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-09-17 00:00:00'
    },
    'dcpro13': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro13/app',
        'total_capital': 3000000,
        'title_prefix': 'Pro13(300WU) PnL',
        'file_suffix': '_dcpro13',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-10-01 00:00:00'
    },
    'dcpro14': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro14/app',
        'total_capital': 4500000,
        'title_prefix': 'Pro14(50BTC) PnL',
        'file_suffix': '_dcpro14',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-10-01 00:00:00'
    },
    'dcpro15': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro15/app',
        'total_capital': 2500000,
        'title_prefix': 'Pro15(250WU) PnL',
        'file_suffix': '_dcpro15',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-10-01 00:00:00'
    },
    'dcpro16': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro16/app',
        'total_capital': 3000000,
        'title_prefix': 'Pro16(300WU) PnL',
        'file_suffix': '_dcpro16',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-10-01 00:00:00'
    },
    'dcpro17': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro17/app',
        'total_capital': 3000000,
        'title_prefix': 'Pro17(300WU) PnL',
        'file_suffix': '_dcpro17',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-10-01 00:00:00'
    },
    # 'dcpmtest3': {
    #     'base_path': '/data_file/CF_data/cf_dc/manager_dcpmtest3/app',
    #     'total_capital': 100000,
    #     'title_prefix': '10W U Test3 Portfolio PnL',
    #     'file_suffix': '_dcpmtest3',
    #     'denominator_ratio': 0.04,
    #     'long_term_start': '2025-07-17 19:00:00'
    # },
    'pmtest2': {
        'base_path': '/data_file/CF_data/cf_dc/manager_maker_dc_pmtest2/app',
        'total_capital': 75000,
        'title_prefix': 'Pmtest2(7.5WU) Bn-bybit PnL',
        'file_suffix': '_pmtest2',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-09-01 00:00:00'
    },
    'pmtest4': {
        'base_path': '/data_file/CF_data/aws_cf_csv/manager_dcpmtest4/app',
        'total_capital': 5000,
        'title_prefix': 'Pmtest4(5000U) Bn-Gate PnL',
        'file_suffix': '_pmtest4',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-10-15 00:00:00'
    },
    'dcpro18': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro18/app',
        'total_capital': 1900000,
        'title_prefix': 'Pro18(634ETH) PnL',
        'file_suffix': '_dcpro18',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-11-15 10:00:00'
    },    
    'dcpro19': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro19/app',
        'total_capital': 1665000,
        'title_prefix': 'Pro19(555ETH) PnL',
        'file_suffix': '_dcpro19',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-11-25 18:00:00'
    },
    'dcpro20': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro20/app',
        'total_capital': 2357550,
        'title_prefix': 'Pro20(26.195BTC) PnL',
        'file_suffix': '_dcpro20',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-11-25 18:00:00'
    },    
    'dcpro21': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro21/app',
        'total_capital': 2160000,
        'title_prefix': 'Pro21(24BTC) PnL',
        'file_suffix': '_dcpro21',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-11-25 18:00:00'
    },    

    'dcpro22': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro22/app',
        'total_capital': 400000,
        'title_prefix': 'Pro22(40WU) PnL',
        'file_suffix': '_dcpro22',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-11-25 18:00:00'
    },    

    'dcpro23': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro23/app',
        'total_capital': 390000,
        'title_prefix': 'Pro23(130ETH) PnL',
        'file_suffix': '_dcpro23',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-11-25 18:00:00'
    },    

    'dcpro24': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro24/app',
        'total_capital': 3870000,
        'title_prefix': 'Pro24(42.955BTC) PnL',
        'file_suffix': '_dcpro24',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-11-28 20:00:00'
    },    
    'dcpro25': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro25/app',
        'total_capital': 1545000,
        'title_prefix': 'Pro25(515ETH) PnL',
        'file_suffix': '_dcpro25',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-12-05 12:00:00'
    },   
    'dcpro26': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro26/app',
        'total_capital': 1800000,
        'title_prefix': 'Pro26(20BTC) PnL',
        'file_suffix': '_dcpro26',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-12-11 18:00:00'
    },   
    'dcpro27': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcpro27/app',
        'total_capital': 720000,
        'title_prefix': 'Pro27(240ETH) PnL',
        'file_suffix': '_dcpro27',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-12-11 18:00:00'
    },   
    'dcbb1': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcbb1/app',
        'total_capital': 1000000,
        'title_prefix': 'Dcbb1(ltp10BTC) PnL',
        'file_suffix': '_dcbb1',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-01-01 04:00:00'
    },   
    'dcbb2': {
        'base_path': '/data_file/CF_data/cf_dc/manager_dcbb2/app',
        'total_capital': 400000,
        'title_prefix': 'Dcbb2(ltp40WU) PnL',
        'file_suffix': '_dcbb2',
        'denominator_ratio': 0.04,
        'long_term_start': '2025-01-01 04:00:00'
    },   
    # 'dcpro28': {
    #     'base_path': '/data_file/CF_data/cf_dc/manager_dcpro28/app',
    #     'total_capital': 1000000,
    #     'title_prefix': 'Pro28(610ETH) PnL',
    #     'file_suffix': '_dcpro28',
    #     'denominator_ratio': 0.04,
    #     'long_term_start': '2025-01-01 04:00:00'
    # },   
    # 'dcpro29': {
    #     'base_path': '/data_file/CF_data/cf_dc/manager_dcpro29/app',
    #     'total_capital': 1000000,
    #     'title_prefix': 'Pro29(10BTC) PnL',
    #     'file_suffix': '_dcpro29',
    #     'denominator_ratio': 0.04,
    #     'long_term_start': '2025-01-01 04:00:00'
    # },   
    # 'dcpro30': {
    #     'base_path': '/data_file/CF_data/cf_dc/manager_dcpro30/app',
    #     'total_capital': 1000000,
    #     'title_prefix': 'Pro30(10BTC) PnL',
    #     'file_suffix': '_dcpro30',
    #     'denominator_ratio': 0.04,
    #     'long_term_start': '2025-01-01 04:00:00'
    # },   
    # 'dcpro31': {
    #     'base_path': '/data_file/CF_data/cf_dc/manager_dcpro31/app',
    #     'total_capital': 1000000,
    #     'title_prefix': 'Pro31(10BTC) PnL',
    #     'file_suffix': '_dcpro31',
    #     'denominator_ratio': 0.04,
    #     'long_term_start': '2025-01-01 04:00:00'
    # },   




}



import hmac
import time
import struct
import base64
import math
from requests.auth import HTTPBasicAuth
import hashlib
import requests

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
    if len(google_code) == 5:
        google_code = '0' + google_code
    return google_code


def get_cookie(_url: str, _username: str, _password: str):
    _response = requests.post(url=_url, auth=HTTPBasicAuth(_username, _password), allow_redirects=False, stream=True)
    print(_response, _response.text, _response.headers)
    _cookies = requests.utils.dict_from_cookiejar(_response.cookies)
    return _cookies

# 返回一个被两重验证的、可以使用的cookie
def login(prefix="mmadminjp"):
    s = requests.Session()
    
    username1 = 'ray_xu' # dcdl用户名
    if prefix == 'mmadminjp3':
        key1 = 'NVCX4PJP32Y4JTJJVCKU5XUQ4M' # dcdl密钥
    elif prefix == 'mmadmin':
        key1 = '7H62KCRCUEOYH2FV4SCYXYMPWE'
    password1 = cal_google_code(key1)
#     print(username1, password1)
    # 如果用http://，会被重定向为https://，可以通过headers得到
    url = f'https://{prefix}.digifinex.org/'
    login_response1 = s.post(url=url, auth=HTTPBasicAuth(username1, password1), allow_redirects=False, stream=True)
#     print(login_response1, login_response1.text, login_response1.headers)
    login_cookies1 = requests.utils.dict_from_cookiejar(login_response1.cookies)
#     print(login_cookies1)
    username2 = 'ray_xu' # dcdl用户名
    key2 = 'HBRDG6BUPJZGC6K7PB2TI3LCOM4XQNDO' # dcdl密钥
    password2 = cal_google_code(key2)
    # 这里必须再登陆一次，否则会重定向到login
    url2 = f'https://{prefix}.digifinex.org/api/login/valid'
#     print(username2, password2, url)
    data = {
        "account":username2,
        "glecode":password2,
    }
    # 不加第一次登陆的cookie会报错
    login_response2 = s.post(url=url2, data = data, headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
                                                                    'Content-Type': 'application/x-www-form-urlencoded'}, 
                                    cookies = login_cookies1, allow_redirects=False)
#     print(login_response2, login_response2.text, login_response2.headers)
    login_cookies2 = requests.utils.dict_from_cookiejar(login_response2.cookies)
    
    return s, login_cookies1