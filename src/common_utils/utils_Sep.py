import pandas as pd
import numpy as np
import re
import CONFIG as C
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
# test02061823pm
if not C.IS_SERVER: #本地才import
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from IPython.display import display


def get_time_col(ex1, ex2):
    """
    根据两个交易所决定要用的时间列（跨所merge用的主时间轴）。
    优先使用OKX的 T 字段，否则默认用 E。
    """
    return 'T' if 'okx' in [ex1,ex2] else 'E'

def rename_columns(columns, prefix):
    output_columns = []

    for c in columns:
        output_columns.append(prefix + c)
            
    return output_columns


def timed(label, collector):
    """Decorator：给函数计时，并把耗时写到 collector[label] 里"""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            res = fn(*args, **kwargs)
            collector[label] = time.perf_counter() - t0
            return res
        return wrapper
    return deco



def parse_start_time(start_str):
    match = re.match(r'^([0-9\-:.\s]+)\s\+0800', str(start_str))
    if match:
        time_str = match.group(1)
        try:
            return pd.Timestamp(time_str)
        except Exception:
            return pd.NaT
    else:
        return pd.NaT

# 读取数据相关

def read_cf_depth(ccy, start_date, end_date, exchange1, market1, exchange2, market2, data_source):
    if data_source == 'inner_win':
        market1_depth_path = f'/Users/rayxu/Desktop/Obentech/dcdlData/{exchange1}/books/{ccy}/{market1}'
        market2_depth_path = f'/Users/rayxu/Desktop/Obentech/dcdlData/{exchange2}/books/{ccy}/{market2}'
    elif data_source == 'outer_ssd':
        market1_depth_path = f'/Volumes/T7/Obentech/dcdlData/{exchange1}/books/{ccy}/{market1}'
        market2_depth_path = f'/Volumes/T7/Obentech/dcdlData/{exchange2}/books/{ccy}/{market2}'
    elif data_source == 'nuts_mm':
        market1_depth_path = f'/Volumes/T7/data/{exchange1}/perp/books/{ccy}'
        market2_depth_path = f'/Volumes/T7/data/{exchange2}/perp/books/{ccy}'
    # if data_source == 'nuts_am_on_mac':
    #     market1_depth_path = f'/Users/rayxu/Downloads/nuts_am/data/{exchange1}/perp/books/{ccy}'
    #     market2_depth_path = f'/Users/rayxu/Downloads/nuts_am/data/{exchange2}/perp/books/{ccy}'
        market1_depth = pd.concat([pd.read_parquet(f'{market1_depth_path}/{ccy}usdt_{dd}_depth5.parquet')
                            for dd in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')])        
        market2_depth = pd.concat([pd.read_parquet(f'{market2_depth_path}/{ccy}usdt_{dd}_depth5.parquet')
                            for dd in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')])
    # else:
    #     market1_depth = pd.concat([pd.read_csv(f'{market1_depth_path}/{ccy}usdt_{dd}_depth5.csv')
    #                       for dd in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')])        
    #     market2_depth = pd.concat([pd.read_csv(f'{market2_depth_path}/{ccy}usdt_{dd}_depth5.csv')
    #                       for dd in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')])

    time_col = get_time_col(exchange1,exchange2)
    market1_depth[time_col] = pd.to_datetime(market1_depth[time_col], unit = 'ms')
    market1_depth.set_index(time_col, inplace=True)
    market1_depth['ws_type'] = "market1_depth"
    market1_depth.drop_duplicates(inplace=True)


    market2_depth[time_col] = pd.to_datetime(market2_depth[time_col], unit = 'ms')
    market2_depth.set_index(time_col, inplace=True)
    market2_depth['ws_type'] = "market2_depth"
    market2_depth.drop_duplicates(inplace=True)
    
    market1_depth.columns = rename_columns(list(market1_depth.columns), 'market1_')
    market2_depth.columns = rename_columns(list(market2_depth.columns), 'market2_')
    ########################################################################################
    # 04-21修改： 改用merge_asof
    # 保证时间索引已经在 column 中（因为 merge_asof 不能用 index 作为 on）
    market1_depth = market1_depth.reset_index()
    market2_depth = market2_depth.reset_index()

    # 使用 merge_asof 精准时间对齐，100ms 容差，向后对齐
    cf_depth = pd.merge_asof(
        market1_depth.sort_values(time_col),
        market2_depth.sort_values(time_col),
        on=time_col,
        direction='backward',
        tolerance=pd.Timedelta('100ms'),
        suffixes=('_market1', '_market2')
    )

    cf_depth[['market1_ws_type', 'market2_ws_type']] = cf_depth[['market1_ws_type', 'market2_ws_type']].fillna('')
    cf_depth['ws_type'] = cf_depth['market1_ws_type'] + cf_depth['market2_ws_type']
    
    cf_depth.dropna(inplace=True)
    cf_depth = cf_depth.fillna(method='ffill').assign(
        sp_open=lambda df: df['market2_bid_price1']-df['market1_ask_price1'],
        sp_close=lambda df: df['market2_ask_price1']-df['market1_bid_price1'],
        sr_open=lambda df: df['sp_open']/df['market1_ask_price1'],
        sr_close=lambda df: df['sp_close']/df['market1_bid_price1'],
        # 用midprice计算
        midprice=lambda df: (df['market1_bid_price1'] + df['market1_ask_price1']) / 2
    )

    # # 计算过去N个tick的收益率（用midprice）
    # for n in [10, 50, 100, 300, 600]:
    #     cf_depth[f'ret_mid_{n/10}s'] = cf_depth['midprice'].pct_change(periods=n)
    #     cf_depth[f'logret_mid_{n/10}s'] = np.log(cf_depth['midprice'] / cf_depth['midprice'].shift(n))
    
    cf_depth.reset_index(inplace=True)
    # cf_depth['received_time_diff_1jump_later'] = cf_depth[time_col].shift(-1) - cf_depth[time_col]
    # cf_depth['received_time_diff_1jump_later'] = cf_depth['received_time_diff_1jump_later'].apply(lambda x:x.total_seconds())
    cf_depth.set_index(time_col, inplace=True)
    beijing_time = cf_depth.index.tz_localize('UTC').tz_convert('Asia/Shanghai').tz_localize(None)
    cf_depth['beijing_time'] = beijing_time
    cf_depth_st_index = cf_depth.index[0]
    cf_depth_et_index = cf_depth.index[-1]
    
    return cf_depth






def plot_bid_price_spread_plotly(cf_depth,
                                 market2_price_col,
                                 market1_price_col='market1_bid_price0',
                                 q_10_bt=None,
                                 q_90_bt=None):
    """
    使用 Plotly 绘制两个市场 bid price 的相对差异（bps），
    并在右侧第二坐标轴上叠加 market1 和 market2 的价格曲线。
    """
    # 计算 spread（bps）
    spread = ((cf_depth[market2_price_col] / cf_depth[market1_price_col]) - 1) * 10000
    spread.index = cf_depth.index

    # 去除极端值
    # lower = spread.quantile(0.001)
    # upper = spread.quantile(0.999)
    # spread_clipped = spread.clip(lower=lower, upper=upper)


    spread_clipped = spread.copy()

    # 分位数
    q10 = spread_clipped.quantile(0.10)
    q90 = spread_clipped.quantile(0.90)

    # 将UTC时间转换为北京时间（UTC+8）
    beijing_time = cf_depth.index.tz_localize('UTC').tz_convert('Asia/Shanghai').tz_localize(None)
    cf_depth['beijing_time'] = beijing_time
    # 创建图
    fig = go.Figure()

    # —— Spread 曲线（左侧 y 轴）——
    fig.add_trace(go.Scatter(
        x=beijing_time,
        y=spread_clipped,
        mode='lines',
        name='Spread (bps)',
        line=dict(color='purple'),
        yaxis='y1'
    ))
    #10% / 90% 分位线
    for val, name, dash_color in [
        (q10, f'10% Quantile: {q10:.2f}', 'red'),
        (q90, f'90% Quantile: {q90:.2f}', 'green'),
        # (q_10_bt, f'10% Quantile (backtest): {q_10_bt:.2f}', 'blue'),
        # (q_90_bt, f'90% Quantile (backtest): {q_90_bt:.2f}', 'orange'),
    ]:
        if val is not None:
            fig.add_trace(go.Scatter(
                x=[beijing_time[0], beijing_time[-1]],
                y=[val, val],
                mode='lines',
                name=name,
                line=dict(color=dash_color, dash='dash'),
                yaxis='y1'
            ))

    # —— 价格曲线（右侧 y 轴）——
    fig.add_trace(go.Scatter(
        x=beijing_time,
        y=cf_depth[market1_price_col],
        mode='lines',
        name=f'{market1_price_col}(binance)',
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=beijing_time,
        y=cf_depth[market2_price_col],
        mode='lines',
        name=f'{market2_price_col}(okx)',
        yaxis='y2'
    ))

    # 布局：定义两个纵轴
    fig.update_layout(
        title=f'Spread & Prices: {market2_price_col} vs {market1_price_col}',
        xaxis=dict(title='Time (Beijing)'),
        yaxis=dict(
            title='Spread (bps)',
            side='left',
            showgrid=False,
        ),
        yaxis2=dict(
            title='Price',
            overlaying='y',
            side='right',
            showgrid=False,
        ),
        hovermode='x unified',
        height=1200,
        width=2400,
        legend=dict(x=0.01, y=0.99)
    )

    fig.show()
    return cf_depth

def plot_bid_price_spread_matplotlib(cf_depth,
                                     market2_price_col,
                                     market1_price_col='market1_bid_price0',
                                     q_10_bt=None,
                                     q_90_bt=None):
    """
    使用 Matplotlib 绘制两个市场 bid price 的相对差异（bps），
    并在右侧第二坐标轴上叠加 market1 和 market2 的价格曲线。
    """
    import matplotlib.pyplot as plt

    # 计算 spread（bps）
    spread = ((cf_depth[market2_price_col] / cf_depth[market1_price_col]) - 1) * 10000
    spread.index = cf_depth.index

    # 去除极端值
    # lower = spread.quantile(0.001)
    # upper = spread.quantile(0.999)
    # spread_clipped = spread.clip(lower=lower, upper=upper)
    spread_clipped = spread.copy()

    # 分位数
    q10 = spread_clipped.quantile(0.10)
    q90 = spread_clipped.quantile(0.90)

    # 将UTC时间转换为北京时间（UTC+8）
    beijing_time = cf_depth.index.tz_localize('UTC').tz_convert('Asia/Shanghai')
    cf_depth['beijing_time'] = beijing_time

    fig, ax1 = plt.subplots(figsize=(24, 12))

    # —— Spread 曲线（左侧 y 轴）——
    ax1.plot(beijing_time, spread_clipped, color='purple', label='Spread (bps)')
    # 10% / 90% 分位线
    for val, name, color in [
        (q10, f'10% Quantile: {q10:.2f}', 'red'),
        (q90, f'90% Quantile: {q90:.2f}', 'green'),
        (q_10_bt, f'10% Quantile (backtest): {q_10_bt:.2f}', 'blue'),
        (q_90_bt, f'90% Quantile (backtest): {q_90_bt:.2f}', 'orange'),
    ]:
        if val is not None:
            ax1.axhline(y=val, color=color, linestyle='--', label=name)

    ax1.set_ylabel('Spread (bps)')
    ax1.set_xlabel('Time (Beijing)')
    ax1.grid(False)

    # —— 价格曲线（右侧 y 轴）——
    ax2 = ax1.twinx()
    ax2.plot(beijing_time, cf_depth[market1_price_col], color='black', label=f'{market1_price_col}(binance)')
    ax2.plot(beijing_time, cf_depth[market2_price_col], color='gray', label=f'{market2_price_col}(okx)')
    ax2.set_ylabel('Price')
    ax2.grid(False)

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f'Spread & Prices: {market2_price_col} vs {market1_price_col}')
    plt.tight_layout()
    plt.show()
    return cf_depth






# 资金费率相关

def process_funding_time(csv_path, exchange, last_time = None):

    ft_exchange_col = {'binance': 'NextFundingTime', 'okx': 'FundingTime', 'bybit': 'FundingTime'}[exchange]
    df = pd.read_csv(csv_path)
    df['FundingTime'] = pd.to_datetime(df[ft_exchange_col], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None) 

    return df

    # if exchange == 'binance':
    #     # 币安的API里面'NextFundingTime'字段是离当前Time最近的下一个结算的时间节点，把NextFundingTime改名为FundingTime来统一命名
    #     df['FundingTime'] = pd.to_datetime(df['NextFundingTime'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    #     df['Time'] = pd.to_datetime(df['Time']).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None) 

    
    # elif exchange == 'okx':
    #     # OKX的API里面'FundingTime'字段才是离当前Time最近的下一个结算的时间节点
    #     df['FundingTime'] = pd.to_datetime(df['FundingTime'],unit='ms',utc = True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)      
    #     df['Time'] = pd.to_datetime(df['Time']).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)

    # elif exchange == 'bybit':
    #     df['FundingTime'] = pd.to_datetime(df['FundingTime'],unit='ms',utc = True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)      
    #     df['Time'] = pd.to_datetime(df['Time']).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)

    # if last_time is not None:
    #     df = df[df["FundingTime"] <= last_time]
    


def process_funding_time_v2(csv_path, exchange: str, last_time=None):
    """
    将 FundingTime 统一为上海本地 naive datetime，并把 Time 转为本地 naive datetime。
    - exchange: 'binance' 使用 NextFundingTime（ms, UTC），'okx'/'bybit' 使用 FundingTime（ms, UTC）
    - last_time: 上海本地 naive datetime；若给定，先用毫秒阈值在 UTC 空间过滤后再做转换
    """
    df = pd.read_csv(csv_path)

    # 选择各交易所应使用的毫秒列（UTC）
    src_ms_col = 'NextFundingTime' if exchange == 'binance' else 'FundingTime'

    # ---- FundingTime: 毫秒(UTC) -> 本地 naive datetime（一次性向量化）----
    ft_ms = df[src_ms_col].to_numpy(np.int64, copy=False)
    ft_local_naive = pd.to_datetime(ft_ms, unit='ms') + pd.Timedelta(hours=8)
    df['FundingTime'] = ft_local_naive.values  # datetime64[ns]（naive，本地时间）

    # ---- Time: 字符串带 +08:00；直接解析并去掉时区即可 ----
    # 对于混合/无时区字符串也兼容；cache=True 常见场景能提速
    t_parsed = pd.to_datetime(df['Time'], errors='coerce', cache=True)

    # 如果是带时区的，优先尝试转到 Asia/Shanghai（万一不是 +08），再去掉 tz；否则直接保留
    if getattr(t_parsed.dtype, "tz", None) is not None:
        try:
            t_parsed = t_parsed.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        except Exception:
            # 多数情况下你的字符串就是 +08:00，直接去 tz 保留本地时刻
            t_parsed = t_parsed.dt.tz_localize(None)

    df['Time'] = t_parsed

    return df


def process_funding_time_v3(csv_path, exchange: str, last_time=None):
    """
    将 FundingTime 统一为上海本地 naive datetime，并把 Time 转为北京时间的datetime（到秒，不保留毫秒）。

    - exchange: 'binance' 使用 NextFundingTime（ms, UTC），'okx'/'bybit' 使用 FundingTime（ms, UTC）
    原因在于：币安的API里面'NextFundingTime'字段是离当前Time最近的下一个结算的时间节点，OKX的API里面'FundingTime'字段才是离当前Time最近的下一个结算的时间节点； 把NextFundingTime改名为FundingTime来统一命名
    - last_time: 上海本地 naive datetime；若给定，先在 UTC 毫秒空间预过滤
    """
    df = pd.read_csv(csv_path)
    if exchange!='gate':
        # 选择各交易所应使用的毫秒列（UTC）
        src_ms_col = 'NextFundingTime' if exchange == 'binance' else 'FundingTime'

        # ---- FundingTime: 毫秒(UTC) -> 本地 naive datetime（一次性向量化，加8小时）----
        ft_ms = df[src_ms_col].to_numpy(np.int64, copy=False)
        df['FundingTime'] = pd.to_datetime(ft_ms, unit='ms') + pd.Timedelta(hours=8)

        # ---- Time: 去掉 +08:00 与小数秒，固定格式解析到秒（最快路径）----
        # 1) 全列转字符串
        s = df['Time'].astype(str)
        # 2) 去掉时区后缀（假设恒为 +08:00）
        s = s.str.replace('+08:00', '', regex=False)
        # 3) 去掉小数秒（.后所有数字）
        s = s.str.replace(r'\.\d+', '', regex=True)
        # 4) 固定格式解析（避免格式推断与时区处理）
        df['Time'] = pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S', errors='coerce', cache=True)
    if exchange == 'gate':
        # 1) 全列转字符串
        s1 = df['Time'].astype(str).str.replace('+08:00', '', regex=False)
        s1 = s1.str.replace(r'\.\d+', '', regex=True)
        df['Time'] = pd.to_datetime(s1, format='%Y-%m-%dT%H:%M:%S', errors='coerce', cache=True)    

        s2 = df['FundingTime'].astype(str).str.replace('+08:00', '', regex=False)
        s2 = s2.str.replace(r'\.\d+', '', regex=True)
        df['FundingTime'] = pd.to_datetime(s2, format='%Y-%m-%dT%H:%M:%S', errors='coerce', cache=True)        

    return df



def analyze_funding_rate_diff_v2(symbol, last_time, lookback_window=14, isPlotMatplotlib=False, isPlotPlotly=False, mode = 'BN-OKX', features_csv_path=None):
    """
    分析单个symbol过去一段时间的资金费率差异
    
    Parameters:
    -----------
    symbol : str, 交易对符号, 格式如'ETH' (不含USDT)
    last_time : datetime, 结束时间
    lookback_window : int 回看天数，默认14天
    isPlot : bool, 是否画图，默认False
    features_csv_path : str, 特征数据CSV文件路径，可选
    
    Returns:
    --------
    dict : 包含分析结果的字典
    {
        'symbol': symbol,
        'latest_cumulative_diff': latest_cumulative_diff,
        'earn': earn,
        'earn_1day': earn_1day,
        'earn_mean': earn_mean,
        'mode_binary_prop': mode_binary_prop,
        'mode_sign_binary': mode_sign_binary,
        'do_indicator': do_indicator,
        'funding_diff': funding_diff,
        'funding_interval_bn': funding_interval_bn,
        'funding_interval_okx': funding_interval_okx
    }
    
    """
    okx_csv     = f'{C.FUNDING_RATE_OKX_DIR}/{symbol}-USDT-SWAP.csv'
    binance_csv = f'{C.FUNDING_RATE_BINANCE_DIR}/{symbol}USDT.csv'
    bybit_csv   = f'{C.FUNDING_RATE_BYBIT_DIR}/{symbol}USDT.csv'
    gate_csv = f'{C.FUNDING_RATE_GATE_DIR}/{symbol}USDT.csv'
    
    from time import perf_counter
    t0 = perf_counter()
    start_time = last_time - pd.Timedelta(days=lookback_window)

    
    if mode == 'BN-OKX':
        df_okx     = process_funding_time_v3(okx_csv, 'okx')
        df_binance = process_funding_time_v3(binance_csv, 'binance')
    elif mode == 'BN-BYBIT':
        df_okx     = process_funding_time_v3(bybit_csv, 'bybit')
        df_binance = process_funding_time_v3(binance_csv, 'binance')
    elif mode == 'BN-GATE':
        df_okx     = process_funding_time_v3(gate_csv, 'gate')
        df_binance = process_funding_time_v3(binance_csv, 'binance')
    else:
        # 默认使用 BN-OKX
        df_okx     = process_funding_time_v3(okx_csv, 'okx')
        df_binance = process_funding_time_v3(binance_csv, 'binance')


    t1 = perf_counter()


    df_b = df_binance[(df_binance['Time'] >= start_time) & (df_binance['Time'] < last_time)].copy()
    df_o = df_okx[(df_okx['Time'] >= start_time) & (df_okx['Time'] < last_time)].copy()

    df_o = df_o.drop_duplicates(subset='FundingTime', keep='last')[:-1] # [:-1]是因为比如我们凌晨四点05获取数据, 那么4:05的数据对应的NextFundingTime应该是08:00的，但08:00的数据还没出来，这个4:05的资费数据实际上无法替代08:00的结算数据，就得去掉。 同样，如果是8h结算的话，最近一次结算时00:00，同样需要drop掉最后一条。
    sum_okx = df_o['FundingRate'].sum()
    df_b = df_b.drop_duplicates(subset='FundingTime', keep='last')[:-1]
    sum_bnb = df_b['FundingRate'].sum()
    earn    = sum_okx - sum_bnb
    day_start = last_time - pd.Timedelta(days=1)

    sum_okx1 = df_o[df_o['Time'] >= day_start]['FundingRate'].sum()
    sum_bnb1 = df_b[df_b['Time'] >= day_start]['FundingRate'].sum()
    earn_1day = sum_okx1 - sum_bnb1

    funding_interval_bn = int((df_b.iloc[-1]['FundingTime'] - df_b.iloc[-2]['FundingTime']).total_seconds() / 3600)
    funding_interval_okx = int((df_o.iloc[-1]['FundingTime'] - df_o.iloc[-2]['FundingTime']).total_seconds() / 3600)

    df_o.rename(columns={'FundingRate': 'FundingRate_okx'}, inplace=True)
    df_b.rename(columns={'FundingRate': 'FundingRate_binance'}, inplace=True)

    if funding_interval_bn == funding_interval_okx:
        funding_diff = df_b[['FundingTime', 'FundingRate_binance']].set_index('FundingTime').join(df_o[['FundingTime', 'FundingRate_okx']].set_index('FundingTime'), how='left')
    elif funding_interval_bn > funding_interval_okx:
        df_o_agg = df_o.set_index('FundingTime').resample(f'{funding_interval_bn}h', label='right', closed='right')['FundingRate_okx'].sum().to_frame()
        funding_diff = df_b[['FundingTime', 'FundingRate_binance']].set_index('FundingTime').join(df_o_agg, how='left')
    else:
        df_b_agg = df_b.set_index('FundingTime').resample(f'{funding_interval_okx}h', label='right', closed='right')['FundingRate_binance'].sum().to_frame()
        funding_diff = df_o[['FundingTime', 'FundingRate_okx']].set_index('FundingTime').join(df_b_agg, how='left')
    
    funding_diff['funding_diff']     = funding_diff['FundingRate_okx'] - funding_diff['FundingRate_binance']


    # 6. 符号众数及占比
    prop      = np.sign(funding_diff['funding_diff']).value_counts(normalize=True)

    prop_full = {k: prop.get(k, 0.0) for k in [-1, 0, 1]}
    if prop_full[1] >= prop_full[-1]:
        mode_sign_binary = 1
        mode_binary_prop = prop_full[1] + prop_full[0]
    else:
        mode_sign_binary = -1
        mode_binary_prop = prop_full[-1] + prop_full[0]

    # 8. 计算按实际跨度的日均收益
    times = df_b['FundingTime'].sort_values()
    if len(times) >= 2:
        total_days = (times.iloc[-1] - times.iloc[0]).total_seconds() / (3600 * 24)
        earn_mean  = earn / total_days if total_days > 0 else np.nan
    else:
        earn_mean = np.nan

    # 7. 指标触发条件
    do_indicator = (
        (mode_binary_prop > 0.75) and
        (mode_sign_binary == np.sign(earn)) and
        (abs(earn_mean) > 0.0002) and
        (abs(earn_1day) > 0.0002) )

    # 保存cumulative diff的最近一期值
    cumsum_values = funding_diff['funding_diff'].cumsum()
    latest_cumulative_diff = cumsum_values.iloc[-1] if len(cumsum_values) > 0 else 0

    t2 = perf_counter()
    print(f"[{symbol}] read_data={t1-t0:.3f}s rest={t2-t1:.3f}s ")
    # 画图部分
    fig_matplotlib = None
    fig_plotly = None
    
    if isPlotMatplotlib:
        import matplotlib.pyplot as plt
        
        # 读取特征信息（如果提供了路径）
        feature_text = None
        if features_csv_path and os.path.exists(features_csv_path):
            try:
                features_df = pd.read_csv(features_csv_path)
                symbol_with_usdt = f'{symbol}-USDT'
                symbol_row = features_df[features_df['Symbol'] == symbol_with_usdt]
                
                if symbol_row.empty:
                    print(f"[Matplotlib] ⚠️  特征文件中未找到 {symbol_with_usdt}")
                else:
                    print(f"[Matplotlib] ✅ 成功读取 {symbol} 的流动性特征信息")
                    feature_cols = [
                        'tick_size_factor', 'q_range', 'do_indicator', 'do_indicator_low', 
                        'do_indicator_MMR', 'ex0_24h_usdt', 'ex1_24h_usdt', 'rank_by_amount', 
                        'market_cap', 'open_interest0', 'open_interest1', 'InsuranceFund0', 
                        'InsuranceFund1', 'IsExtremeSpread', 'IsExtremeFr', 'IsExtremeFrSpread'
                    ]
                    
                    # 构建显示文本
                    feature_lines = [f'Liquidity Info ({symbol_with_usdt})']
                    feature_lines.append('-' * 35)
                    
                    # 需要转换为万单位的字段
                    wan_unit_cols = ['ex0_24h_usdt', 'ex1_24h_usdt', 'open_interest0', 
                                     'open_interest1', 'InsuranceFund0', 'InsuranceFund1', 'market_cap']
                    
                    for col in feature_cols:
                        if col in symbol_row.columns:
                            value = symbol_row[col].iloc[0]
                            if pd.notna(value):
                                # 检查是否需要转换为万单位
                                if col in wan_unit_cols:
                                    try:
                                        wan_value = int(float(value) / 10000)
                                        formatted_value = f"{wan_value}(WU)"
                                    except (ValueError, TypeError):
                                        formatted_value = "N/A"
                                elif isinstance(value, (int, np.integer)):
                                    formatted_value = f"{value:,}"
                                elif isinstance(value, (float, np.floating)):
                                    if abs(value) >= 1000:
                                        formatted_value = f"{value:,.2f}"
                                    else:
                                        formatted_value = f"{value:.4g}"
                                else:
                                    formatted_value = str(value)
                            else:
                                formatted_value = "N/A"
                            feature_lines.append(f"{col}: {formatted_value}")
                    
                    feature_text = '\n'.join(feature_lines)
            except Exception as e:
                print(f"[Matplotlib] ❌ 读取特征文件时出错: {e}")
        elif features_csv_path and not os.path.exists(features_csv_path):
            print(f"[Matplotlib] ⚠️  特征文件不存在: {features_csv_path}")
        elif not features_csv_path:
            print(f"[Matplotlib] ℹ️  未提供特征文件路径，跳过流动性信息展示")
        
        # 根据是否有特征信息调整图表大小和布局
        if feature_text:
            # 有特征信息：使用更宽的画布，并添加右侧文本区域
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(2, 2, figure=fig, width_ratios=[3, 1], hspace=0.15, wspace=0.08,
                         left=0.08, right=0.98, top=0.95, bottom=0.08)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax_text = fig.add_subplot(gs[:, 1])
        else:
            # 无特征信息：使用原来的布局
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot the funding rate difference in the first subplot
        (funding_diff['funding_diff']*10000).plot(ax=ax1, color='blue', linewidth=1.5)
        if mode == 'BN-OKX':
            ax1.set_title(f'OKX - Binance Funding Rate Difference for {symbol}', fontsize=14, fontweight='bold')
        elif mode == 'BN-BYBIT':
            ax1.set_title(f'Bybit - Binance Funding Rate Difference for {symbol}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Difference (bps)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot the cumulative funding rate difference in the second subplot
        # If the cumulative sum is negative, invert it
        if cumsum_values.iloc[-1] < 0:
            cumsum_values = -cumsum_values
        (cumsum_values*10000).plot(ax=ax2, color='green', linewidth=1.5)
        ax2.set_title(f'Cumulative Funding Rate Difference for {symbol}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('FundingTime', fontsize=12)
        ax2.set_ylabel('Cumulative Difference (bps)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 如果有特征信息，在右侧显示
        if feature_text:
            ax_text.axis('off')
            # 使用填充整个区域的文本框，与左侧图表高度对齐
            ax_text.text(0.02, 0.99, feature_text, 
                        transform=ax_text.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        horizontalalignment='left',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                                 alpha=0.5, edgecolor='gray', linewidth=0.5))

        # Improve overall appearance (不使用tight_layout，因为我们已经手动设置了边距)
        fig_matplotlib = fig
        plt.show()


    if isPlotPlotly:


        # 1) diff(bps)
        diff_bps = (funding_diff["funding_diff"] * 10000)

        # 2) cumulative(bps)，若最后为负则翻正
        cumsum_plot = cumsum_values.copy()
        if len(cumsum_plot) > 0 and cumsum_plot.iloc[-1] < 0:
            cumsum_plot = -cumsum_plot
        cumsum_bps = (cumsum_plot * 10000)

        if mode == 'BN-OKX':
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=(
                    f"OKX - Binance Funding Rate Difference for {symbol} (bps)",
                    f"Cumulative Funding Rate Difference for {symbol} (bps)"
                )
            )
        elif mode == 'BN-BYBIT':
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=(
                    f"Bybit - Binance Funding Rate Difference for {symbol} (bps)",
                    f"Cumulative Funding Rate Difference for {symbol} (bps)"
                )
            )


        # 上图：单期 diff
        fig.add_trace(
            go.Scatter(
                x=funding_diff.index,
                y=diff_bps.values,
                mode="lines",
                name="funding_diff (bps)",
            ),
            row=1, col=1
        )

        # 下图：累计 diff
        fig.add_trace(
            go.Scatter(
                x=cumsum_bps.index,
                y=cumsum_bps.values,
                mode="lines",
                name="cumulative (bps)",
            ),
            row=2, col=1
        )

        fig.update_yaxes(title_text="Difference (bps)", row=1, col=1, zeroline=True)
        fig.update_yaxes(title_text="Cumulative (bps)", row=2, col=1, zeroline=True)

        fig.update_layout(
            title=f"Funding Rate Diff (OKX - Binance) — {symbol}",
            height=600,
            width= 800, 
            hovermode="x unified",
            showlegend=True,
            margin=dict(l=40, r=20, t=80, b=40),
        )

        # 读取并显示特征信息
        if features_csv_path and os.path.exists(features_csv_path):
            try:
                features_df = pd.read_csv(features_csv_path)
                symbol_with_usdt = f'{symbol}-USDT'
                
                # 查找对应的Symbol行
                symbol_row = features_df[features_df['Symbol'] == symbol_with_usdt]
                
                if not symbol_row.empty:
                    print(f"[Plotly] ✅ 成功读取 {symbol} 的流动性特征信息")
                else:
                    print(f"[Plotly] ⚠️  特征文件中未找到 {symbol_with_usdt}")
                    # 提取需要的特征列
                    feature_cols = [
                        'tick_size_factor', 'q_range', 'do_indicator', 'do_indicator_low', 
                        'do_indicator_MMR', 'ex0_24h_usdt', 'ex1_24h_usdt', 'rank_by_amount', 
                        'market_cap', 'open_interest0', 'open_interest1', 'InsuranceFund0', 
                        'InsuranceFund1', 'IsExtremeSpread', 'IsExtremeFr', 'IsExtremeFrSpread'
                    ]
                    
                    # 构建显示文本
                    feature_text = f"<b>Liquidity Info ({symbol_with_usdt})</b><br>"
                    
                    # 需要转换为万单位的字段
                    wan_unit_cols = ['ex0_24h_usdt', 'ex1_24h_usdt', 'open_interest0', 
                                     'open_interest1', 'InsuranceFund0', 'InsuranceFund1', 'market_cap']
                    
                    for col in feature_cols:
                        if col in symbol_row.columns:
                            value = symbol_row[col].iloc[0]
                            # 格式化数值
                            if pd.notna(value):
                                # 检查是否需要转换为万单位
                                if col in wan_unit_cols:
                                    try:
                                        wan_value = int(float(value) / 10000)
                                        formatted_value = f"{wan_value}(WU)"
                                    except (ValueError, TypeError):
                                        formatted_value = "N/A"
                                elif isinstance(value, (int, np.integer)):
                                    formatted_value = f"{value:,}"
                                elif isinstance(value, (float, np.floating)):
                                    if abs(value) >= 1000:
                                        formatted_value = f"{value:,.2f}"
                                    else:
                                        formatted_value = f"{value:.4g}"
                                else:
                                    formatted_value = str(value)
                            else:
                                formatted_value = "N/A"
                            feature_text += f"<br>{col}: {formatted_value}"
                    
                    # 添加注释框
                    fig.add_annotation(
                        text=feature_text,
                        xref="paper", yref="paper",
                        x=1.02, y=0.98,
                        xanchor="left", yanchor="top",
                        align="left",
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=9, family="monospace"),
                    )
                    
                    # 调整布局以容纳注释
                    fig.update_layout(
                        width=1200,  # 增加宽度以容纳特征信息
                        margin=dict(l=40, r=350, t=80, b=40),  # 增加右边距
                    )
            except Exception as e:
                print(f"[Plotly] ❌ 读取特征文件时出错: {e}")
        elif features_csv_path and not os.path.exists(features_csv_path):
            print(f"[Plotly] ⚠️  特征文件不存在: {features_csv_path}")
        elif not features_csv_path:
            print(f"[Plotly] ℹ️  未提供特征文件路径，跳过流动性信息展示")

        fig_plotly = fig
        fig.show()



    return {
        'symbol': symbol,
        'latest_cumulative_diff': latest_cumulative_diff,
        'earn': earn,
        'earn_1day': earn_1day,
        'earn_mean': earn_mean,
        'mode_binary_prop': mode_binary_prop,
        'mode_sign_binary': mode_sign_binary,
        'do_indicator': do_indicator,
        'funding_diff': funding_diff,
        'funding_interval_bn': funding_interval_bn,
        'funding_interval_okx': funding_interval_okx,
        'fig_matplotlib': fig_matplotlib,
        'fig_plotly': fig_plotly
    }



def analyze_funding_rate_single_v1(
    symbol,
    last_time,
    lookback_window=14,
    isPlotMatplotlib=False,
    isPlotPlotly=False,
    mode='BN',  # 'BN'/'OKX'/'BYBIT'/'GATE'
    features_csv_path=None,

    # 触发阈值（你可以按自己口味调）
    min_mode_prop=0.75,     # funding 为正(含0)的占比门槛
    min_earn_mean=0.0002,   # 日均 funding(小数)门槛，0.0002=2bps/day
    min_earn_1day=0.0002,   # 最近1天累计 funding 门槛
    ):
    """
    单交易所资金费率分析：用于筛 long spot & short perp（要求 funding > 0 且稳定）

    Returns dict:
    {
        'symbol': symbol,
        'exchange': mode,
        'latest_cumulative': latest_cumulative,
        'earn': earn,                 # 近N天累计 funding
        'earn_1day': earn_1day,       # 近1天累计 funding
        'earn_mean': earn_mean,       # 日均 funding
        'mode_binary_prop': mode_binary_prop,  # 众数符号(+1或-1)占比(含0)
        'mode_sign_binary': mode_sign_binary,  # +1 或 -1
        'do_indicator': do_indicator,
        'funding_df': funding_df,     # index=FundingTime, col=FundingRate
        'funding_interval': funding_interval, # 小时
        'fig_matplotlib': fig_matplotlib,
        'fig_plotly': fig_plotly,
    }
    """

    start_time = last_time - pd.Timedelta(days=lookback_window)

    # --- 1) 路径与交易所映射（沿用你原来的 C 常量目录）---
    mode_up = mode.upper()
    if mode_up == 'BN':
        csv_path = f'{C.FUNDING_RATE_BINANCE_DIR}/{symbol}USDT.csv'
        exch = 'binance'
        title_exch = 'Binance'
    elif mode_up == 'OKX':
        csv_path = f'{C.FUNDING_RATE_OKX_DIR}/{symbol}-USDT-SWAP.csv'
        exch = 'okx'
        title_exch = 'OKX'
    elif mode_up == 'BYBIT':
        csv_path = f'{C.FUNDING_RATE_BYBIT_DIR}/{symbol}USDT.csv'
        exch = 'bybit'
        title_exch = 'Bybit'
    elif mode_up == 'GATE':
        csv_path = f'{C.FUNDING_RATE_GATE_DIR}/{symbol}_USDT.csv'
        exch = 'gate'
        title_exch = 'Gate'
    else:
        csv_path = f'{C.FUNDING_RATE_BINANCE_DIR}/{symbol}USDT.csv'
        exch = 'binance'
        title_exch = 'Binance'

    # print(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[{symbol}] funding csv not found: {csv_path}")

    # --- 2) 读 & 统一时间（复用你的函数）---
    df = process_funding_time_v3(csv_path, exch)

    # --- 3) 时间窗过滤 ---
    d = df[(df['Time'] >= start_time) & (df['Time'] < last_time)].copy()

    # --- 4) 去重 + 丢最后一条（与你 diff 版本一致）---
    d = d.drop_duplicates(subset='FundingTime', keep='last')
    if len(d) >= 1:
        d = d[:-1]  # drop last incomplete funding bucket

    fig_matplotlib = None
    fig_plotly = None

    if len(d) < 2:
        return {
            'symbol': symbol,
            'exchange': mode_up,
            'latest_cumulative': 0.0,
            'earn': 0.0,
            'earn_1day': 0.0,
            'earn_mean': np.nan,
            'mode_binary_prop': 0.0,
            'mode_sign_binary': 0,
            'do_indicator': False,
            'funding_df': pd.DataFrame(),
            'funding_interval': np.nan,
            'fig_matplotlib': fig_matplotlib,
            'fig_plotly': fig_plotly,
            'error': 'insufficient_data'
        }

    # --- 5) 计算 funding 指标 ---
    earn = float(d['FundingRate'].sum())

    day_start = last_time - pd.Timedelta(days=1)
    earn_1day = float(d[d['Time'] >= day_start]['FundingRate'].sum())

    funding_interval = int((d.iloc[-1]['FundingTime'] - d.iloc[-2]['FundingTime']).total_seconds() / 3600)

    times = d['FundingTime'].sort_values()
    total_days = (times.iloc[-1] - times.iloc[0]).total_seconds() / (3600 * 24)
    earn_mean = (earn / total_days) if total_days > 0 else np.nan

    # --- 6) 符号众数及占比（复刻你原逻辑：0 也计入）---
    sgn = np.sign(d['FundingRate'].to_numpy())
    prop = pd.Series(sgn).value_counts(normalize=True)
    prop_full = {k: prop.get(k, 0.0) for k in [-1, 0, 1]}

    if prop_full[1] >= prop_full[-1]:
        mode_sign_binary = 1
        mode_binary_prop = prop_full[1] + prop_full[0]
    else:
        mode_sign_binary = -1
        mode_binary_prop = prop_full[-1] + prop_full[0]

    # --- 7) 触发条件：吃正 funding（long spot & short perp）---
    do_indicator = (
        (mode_binary_prop > min_mode_prop) and
        (mode_sign_binary == 1) and
        (earn_mean > min_earn_mean) and
        (earn_1day > min_earn_1day)
    )

    funding_df = d[['FundingTime', 'FundingRate']].set_index('FundingTime').sort_index()
    cumsum_values = funding_df['FundingRate'].cumsum()
    latest_cumulative = float(cumsum_values.iloc[-1]) if len(cumsum_values) > 0 else 0.0

    # =========================
    # Plotting (Matplotlib)
    # =========================
    if isPlotMatplotlib:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # 读取特征信息（如果提供了路径）
        feature_text = None
        if features_csv_path and os.path.exists(features_csv_path):
            try:
                features_df = pd.read_csv(features_csv_path)
                symbol_with_usdt = f'{symbol}-USDT'
                symbol_row = features_df[features_df['Symbol'] == symbol_with_usdt]

                if symbol_row.empty:
                    print(f"[Matplotlib] ⚠️  特征文件中未找到 {symbol_with_usdt}")
                else:
                    print(f"[Matplotlib] ✅ 成功读取 {symbol} 的流动性特征信息")
                    feature_cols = [
                        'tick_size_factor', 'q_range', 'do_indicator', 'do_indicator_low',
                        'do_indicator_MMR', 'ex0_24h_usdt', 'ex1_24h_usdt', 'rank_by_amount',
                        'market_cap', 'open_interest0', 'open_interest1', 'InsuranceFund0',
                        'InsuranceFund1', 'IsExtremeSpread', 'IsExtremeFr', 'IsExtremeFrSpread'
                    ]

                    feature_lines = [f'Liquidity Info ({symbol_with_usdt})', '-' * 35]
                    wan_unit_cols = [
                        'ex0_24h_usdt', 'ex1_24h_usdt', 'open_interest0', 'open_interest1',
                        'InsuranceFund0', 'InsuranceFund1', 'market_cap'
                    ]

                    for col in feature_cols:
                        if col in symbol_row.columns:
                            value = symbol_row[col].iloc[0]
                            if pd.notna(value):
                                if col in wan_unit_cols:
                                    try:
                                        wan_value = int(float(value) / 10000)
                                        formatted_value = f"{wan_value}(WU)"
                                    except (ValueError, TypeError):
                                        formatted_value = "N/A"
                                elif isinstance(value, (int, np.integer)):
                                    formatted_value = f"{value:,}"
                                elif isinstance(value, (float, np.floating)):
                                    formatted_value = f"{value:,.4g}"
                                else:
                                    formatted_value = str(value)
                            else:
                                formatted_value = "N/A"
                            feature_lines.append(f"{col}: {formatted_value}")

                    feature_text = '\n'.join(feature_lines)
            except Exception as e:
                print(f"[Matplotlib] ❌ 读取特征文件时出错: {e}")
        elif features_csv_path and not os.path.exists(features_csv_path):
            print(f"[Matplotlib] ⚠️  特征文件不存在: {features_csv_path}")

        # 画布布局
        if feature_text:
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(
                2, 2, figure=fig,
                width_ratios=[3, 1],
                hspace=0.15, wspace=0.08,
                left=0.08, right=0.98, top=0.95, bottom=0.08
            )
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax_text = fig.add_subplot(gs[:, 1])
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 上图：单期 funding bps
        (funding_df['FundingRate'] * 10000).plot(ax=ax1, color='blue', linewidth=1.5)
        ax1.set_title(f'{title_exch} Funding Rate for {symbol} (bps)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Funding (bps)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 下图：累计 funding bps（如果最后为负，翻正展示“吃费强度”的幅度，保持你原风格）
        cumsum_plot = cumsum_values.copy()
        if len(cumsum_plot) > 0 and cumsum_plot.iloc[-1] < 0:
            cumsum_plot = -cumsum_plot
        (cumsum_plot * 10000).plot(ax=ax2, color='green', linewidth=1.5)
        ax2.set_title(f'Cumulative Funding for {symbol} (bps)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('FundingTime', fontsize=12)
        ax2.set_ylabel('Cumulative (bps)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 右侧特征框
        if feature_text:
            ax_text.axis('off')
            ax_text.text(
                0.02, 0.99, feature_text,
                transform=ax_text.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left',
                fontfamily='monospace',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='wheat', alpha=0.5,
                    edgecolor='gray', linewidth=0.5
                )
            )

        fig_matplotlib = fig
        plt.show()

    # =========================
    # Plotting (Plotly)
    # =========================
    if isPlotPlotly:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 单期 bps
        fr_bps = funding_df['FundingRate'] * 10000

        # 累计 bps（保持你原风格：最后为负则翻正）
        cumsum_plot = cumsum_values.copy()
        if len(cumsum_plot) > 0 and cumsum_plot.iloc[-1] < 0:
            cumsum_plot = -cumsum_plot
        cumsum_bps = cumsum_plot * 10000

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(
                f"{title_exch} Funding Rate for {symbol} (bps)",
                f"Cumulative Funding for {symbol} (bps)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=fr_bps.index,
                y=fr_bps.values,
                mode="lines",
                name="funding (bps)",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=cumsum_bps.index,
                y=cumsum_bps.values,
                mode="lines",
                name="cumulative (bps)",
            ),
            row=2, col=1
        )

        fig.update_yaxes(title_text="Funding (bps)", row=1, col=1, zeroline=True)
        fig.update_yaxes(title_text="Cumulative (bps)", row=2, col=1, zeroline=True)

        fig.update_layout(
            title=f"{title_exch} Funding — {symbol}",
            height=600,
            width=800,
            hovermode="x unified",
            showlegend=True,
            margin=dict(l=40, r=20, t=80, b=40),
        )

        # 读取并显示特征信息（右侧注释）
        if features_csv_path and os.path.exists(features_csv_path):
            try:
                features_df = pd.read_csv(features_csv_path)
                symbol_with_usdt = f'{symbol}-USDT'
                symbol_row = features_df[features_df['Symbol'] == symbol_with_usdt]

                if symbol_row.empty:
                    print(f"[Plotly] ⚠️  特征文件中未找到 {symbol_with_usdt}")
                else:
                    print(f"[Plotly] ✅ 成功读取 {symbol} 的流动性特征信息")

                    feature_cols = [
                        'tick_size_factor', 'q_range', 'do_indicator', 'do_indicator_low',
                        'do_indicator_MMR', 'ex0_24h_usdt', 'ex1_24h_usdt', 'rank_by_amount',
                        'market_cap', 'open_interest0', 'open_interest1', 'InsuranceFund0',
                        'InsuranceFund1', 'IsExtremeSpread', 'IsExtremeFr', 'IsExtremeFrSpread'
                    ]
                    wan_unit_cols = [
                        'ex0_24h_usdt', 'ex1_24h_usdt', 'open_interest0', 'open_interest1',
                        'InsuranceFund0', 'InsuranceFund1', 'market_cap'
                    ]

                    feature_text = f"<b>Liquidity Info ({symbol_with_usdt})</b><br>"
                    feature_text += "<br>" + "-" * 35 + "<br>"

                    for col in feature_cols:
                        if col in symbol_row.columns:
                            value = symbol_row[col].iloc[0]
                            if pd.notna(value):
                                if col in wan_unit_cols:
                                    try:
                                        wan_value = int(float(value) / 10000)
                                        formatted_value = f"{wan_value}(WU)"
                                    except (ValueError, TypeError):
                                        formatted_value = "N/A"
                                elif isinstance(value, (int, np.integer)):
                                    formatted_value = f"{value:,}"
                                elif isinstance(value, (float, np.floating)):
                                    formatted_value = f"{value:,.4g}"
                                else:
                                    formatted_value = str(value)
                            else:
                                formatted_value = "N/A"
                            feature_text += f"<br>{col}: {formatted_value}"

                    fig.add_annotation(
                        text=feature_text,
                        xref="paper", yref="paper",
                        x=1.02, y=0.98,
                        xanchor="left", yanchor="top",
                        align="left",
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=9, family="monospace"),
                    )

                    fig.update_layout(
                        width=1200,
                        margin=dict(l=40, r=350, t=80, b=40),
                    )
            except Exception as e:
                print(f"[Plotly] ❌ 读取特征文件时出错: {e}")
        elif features_csv_path and not os.path.exists(features_csv_path):
            print(f"[Plotly] ⚠️  特征文件不存在: {features_csv_path}")

        fig_plotly = fig
        fig.show()

    return {
        'symbol': symbol,
        'exchange': mode_up,
        'latest_cumulative': latest_cumulative,
        'earn': earn,
        'earn_1day': earn_1day,
        'earn_mean': earn_mean,
        'mode_binary_prop': mode_binary_prop,
        'mode_sign_binary': mode_sign_binary,
        'do_indicator': do_indicator,
        'funding_df': funding_df,
        'funding_interval': funding_interval,
        'fig_matplotlib': fig_matplotlib,
        'fig_plotly': fig_plotly
    }




def load_spread_data_history_data(symbol: str, st, et, mode='BN-OKX'):
    """
    返回：
      cf_depth_resampled: 含市场价格等的 60s 采样数据
      spread_bid, spread_ask: 两条 resampled 的价差序列（单位：bps）
    依赖：
      read_cf_depth(symbol, st, et, exchange1, market1, exchange2, market2, data_source)
      其中需要输出有 beijing_time 列或 index 可设为 beijing_time
    
    Parameters:
    -----------
    symbol : str, 交易对符号
    st : datetime, 开始时间
    et : datetime, 结束时间
    mode : str, 交易所模式，如 'BN-OKX', 'BN-BYBIT', 'BN-GATE' 等
    """
    ccy = symbol
    
    # 根据 mode 确定深度文件名称
    # 文件命名格式: depth_{exchange2}_{exchange1}_{symbol}-usdt_1min.csv
    mode_to_file_pattern = {
        'BN-OKX': 'depth_okx_binance',
        'BN-BYBIT': 'depth_bybit5_binance',
        'BN-GATE': 'depth_gate_binance',
    }
    
    file_pattern = mode_to_file_pattern.get(mode, 'depth_okx_binance')
    
    # 读聚合后的深度数据
    depth_path = f'{C.MINUTE_DEPTH_DIR}/{file_pattern}_{symbol.lower()}-usdt_1min.csv'
    if C.IS_SERVER:
        depth_path = f'{C.MINUTE_DEPTH_DIR}/{file_pattern}_{symbol.lower()}-usdt_1min.csv'
    
    cf_depth = pd.read_csv(depth_path)
    cf_depth['event_time'] = pd.to_datetime(cf_depth['event_time'], unit='ms')
    cf_depth['beijing_time'] = cf_depth['event_time']+pd.Timedelta(hours=8)
    cf_depth = cf_depth[(cf_depth['beijing_time']>=pd.to_datetime(st)) & (cf_depth['beijing_time']<=pd.to_datetime(et))]
    cf_depth = cf_depth.drop_duplicates(subset='event_time', keep='last')
    cf_depth.set_index('beijing_time', inplace=True)
    # 增加最后一个时间点00:00:00（之前是到23:59:59）
    st = pd.to_datetime(st)
    et = pd.to_datetime(et)
    end_midnight = et + pd.Timedelta(days=1) 

    spread_bid = cf_depth['avg_sr_bid']
    spread_ask = cf_depth['avg_sr_ask']
    return cf_depth, spread_bid, spread_ask


def load_funding_diffs(symbol: str, last_time, spread_bid: pd.Series, spread_ask: pd.Series,
                       lookback_days=4, mode='BN-OKX'):
    """
    返回：
      funding_diff: DataFrame(index=FundingTime, cols=['FundingRate_okx','FundingRate_binance','funding_diff'])
      merged_df:    含 Type1 结果（funding_diff_adj_cumsum）
      merged_df2:   含 Type2 结果（funding_diff_signed_cumsum）
    依赖：
      process_funding_time_v3(csv_path, exchange)
      你的资金费率 CSV 路径结构
      
    Parameters:
    -----------
    symbol : str, 交易对符号
    last_time : datetime, 结束时间
    spread_bid : pd.Series, bid价差序列
    spread_ask : pd.Series, ask价差序列
    lookback_days : int, 回看天数，默认4天
    mode : str, 交易所模式，如 'BN-OKX', 'BN-BYBIT', 'BN-GATE' 等
    """
    symbol = symbol.upper()
    
    # 根据 mode 选择对应的交易所和文件路径
    if mode == 'BN-OKX':
        exchange2_csv = f"{C.FUNDING_RATE_OKX_DIR}/{symbol}-USDT-SWAP.csv"
        exchange1_csv = f"{C.FUNDING_RATE_BINANCE_DIR}/{symbol}USDT.csv"
        exchange2_name = 'okx'
        exchange1_name = 'binance'
    elif mode == 'BN-BYBIT':
        exchange2_csv = f"{C.FUNDING_RATE_BYBIT_DIR}/{symbol}USDT.csv"
        exchange1_csv = f"{C.FUNDING_RATE_BINANCE_DIR}/{symbol}USDT.csv"
        exchange2_name = 'bybit'
        exchange1_name = 'binance'
    elif mode == 'BN-GATE':
        exchange2_csv = f"{C.FUNDING_RATE_GATE_DIR}/{symbol}USDT.csv"
        exchange1_csv = f"{C.FUNDING_RATE_BINANCE_DIR}/{symbol}USDT.csv"
        exchange2_name = 'gate'
        exchange1_name = 'binance'
    else:
        # 默认使用 BN-OKX
        exchange2_csv = f"{C.FUNDING_RATE_OKX_DIR}/{symbol}-USDT-SWAP.csv"
        exchange1_csv = f"{C.FUNDING_RATE_BINANCE_DIR}/{symbol}USDT.csv"
        exchange2_name = 'okx'
        exchange1_name = 'binance'

    start_time = last_time - pd.Timedelta(days=lookback_days)

    df_okx     = process_funding_time_v3(exchange2_csv, exchange2_name)
    df_binance = process_funding_time_v3(exchange1_csv, exchange1_name)

    df_b = df_binance[(df_binance['Time'] >= start_time) & (df_binance['Time'] < last_time)].copy()
    df_o = df_okx[(df_okx['Time'] >= start_time) & (df_okx['Time'] < last_time)].copy()

    # 去重，仅保留每个 FundingTime 最后一条（并去掉最后一期未结算点）
    df_o = df_o.drop_duplicates(subset='FundingTime', keep='last')[:-1]
    df_b = df_b.drop_duplicates(subset='FundingTime', keep='last')[:-1]


    # 对齐不同结算周期
    funding_interval_bn  = int((df_b.iloc[-1]['FundingTime'] - df_b.iloc[-2]['FundingTime']).total_seconds() / 3600)
    funding_interval_okx = int((df_o.iloc[-1]['FundingTime'] - df_o.iloc[-2]['FundingTime']).total_seconds() / 3600)

    df_o = df_o.rename(columns={'FundingRate': 'FundingRate_okx'})
    df_b = df_b.rename(columns={'FundingRate': 'FundingRate_binance'})

    if funding_interval_bn == funding_interval_okx:
        funding_diff = (
            df_b[['FundingTime', 'FundingRate_binance']].set_index('FundingTime')
            .join(df_o[['FundingTime', 'FundingRate_okx']].set_index('FundingTime'), how='left')
        )
    elif funding_interval_bn > funding_interval_okx:
        df_o_agg = (
            df_o.set_index('FundingTime')
               .resample(f'{funding_interval_bn}h', label='right', closed='right')['FundingRate_okx']
               .sum().to_frame()
        )
        funding_diff = df_b[['FundingTime', 'FundingRate_binance']].set_index('FundingTime').join(df_o_agg, how='left')
    else:
        df_b_agg = (
            df_b.set_index('FundingTime')
               .resample(f'{funding_interval_okx}h', label='right', closed='right')['FundingRate_binance']
               .sum().to_frame()
        )
        funding_diff = df_o[['FundingTime', 'FundingRate_okx']].set_index('FundingTime').join(df_b_agg, how='left')

    funding_diff['funding_diff'] = funding_diff['FundingRate_okx'] - funding_diff['FundingRate_binance']


    # ---------- Type1：中位数规则 ----------
    spread_resampled = spread_bid.copy().to_frame()
    spread_resampled.columns = ['spread_bid_resampled']
    spread_resampled['spread_ask_resampled'] = spread_ask

    merged_df = pd.merge_asof(funding_diff, spread_resampled, left_index=True, right_index=True,tolerance=pd.Timedelta(seconds=90),direction='nearest')
    med_bid = spread_bid.quantile(0.5)
    med_ask = spread_ask.quantile(0.5) 

    merged_df['sign'] = np.where(
        merged_df['spread_ask_resampled'] > med_ask,  1,
        np.where(merged_df['spread_bid_resampled'] < med_bid, -1, 1)
    )
    merged_df['funding_diff_adj'] = merged_df['funding_diff'] * merged_df['sign']
    merged_df['funding_diff_adj_cumsum'] = merged_df['funding_diff_adj'].cumsum()

    # ---------- Type2：p40 / p60 规则 ----------
    merged_df2 = merged_df.copy()
    fd      = merged_df2['funding_diff']
    spr_bid = merged_df2['spread_bid_resampled']
    spr_ask = merged_df2['spread_ask_resampled']

    p40 = spread_bid.quantile(0.1)
    p60 = spread_ask.quantile(0.9)
    merged_df2['p40'] = p40
    merged_df2['p60'] = p60
    merged_df2['bid_50'] = med_bid
    merged_df2['ask_50'] = med_ask
    merged_df2['sign_rule'] = np.where(
        ((fd >= 0) & (spr_bid >= p40)) | ((fd <= 0) & (spr_ask >= p60)),
        1,
        np.where(fd == 0, 0, -1)
    ).astype('int8')

    merged_df2['funding_diff_signed'] = fd * merged_df2['sign_rule']
    merged_df2['funding_diff_signed_cumsum'] = merged_df2['funding_diff_signed'].cumsum()

    return funding_diff, merged_df, merged_df2


def plot_cf_panels(symbol: str, start_time, last_time,
                   spread_bid: pd.Series, bid_px: pd.Series,
                   funding_diff: pd.DataFrame,
                   merged_df: pd.DataFrame, merged_df2: pd.DataFrame):
    """
    生成一个大图：上面三轴（Spread/Funding/Bid），下面 2×2（FundingDiff、Rule-Median累计、Cumulative累计、p40/p60累计）
    返回 fig
    """
    # 颜色
    COLOR_SPREAD  = "#66C2A5"
    COLOR_FUNDING = "#FC8D62"
    COLOR_BID     = "#8DA0CB"
    C1 = "#0072B2"; C2 = "#009E73"; C3 = "#E69F00"; C4 = "#D55E00"

    idx = spread_bid.index
    funding_sorted = funding_diff.sort_index()

    series_fd   = (funding_diff['funding_diff'] * 10000).dropna()
    cumsum_vals = funding_diff['funding_diff'].cumsum()
    if cumsum_vals.iloc[-1] < 0:
        cumsum_vals = -cumsum_vals
    series_cum  = (cumsum_vals * 10000).dropna()
    series_adj  = (merged_df['funding_diff_adj_cumsum'] * 10000).dropna()
    series_sig  = (merged_df2['funding_diff_signed_cumsum'] * 10000).dropna()

    fig = plt.figure(constrained_layout=True, figsize=(24, 16))
    gs  = GridSpec(nrows=3, ncols=2, figure=fig, height_ratios=[1.4, 1, 1])

    # 顶部三轴
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.patch.set_visible(False)
    ax_top.grid(False)

    l1, = ax_top.plot(idx, spread_bid.values, label='Spread',
                      color=COLOR_SPREAD, linewidth=1.6, zorder=3)
    ax_top.set_ylabel('Spread')
    ax_top.set_xlabel('Time')

    ax_top_r = ax_top.twinx()
    ax_top_r.patch.set_visible(False)
    l2, = ax_top_r.plot(funding_sorted.index, funding_sorted['funding_diff'].values,
                        linestyle='-', linewidth=1.8, color=COLOR_FUNDING,
                        label='Funding Diff', zorder=2)
    ax_top_r.set_ylabel('Funding Diff')

    ax_top_r2 = ax_top.twinx()
    ax_top_r2.spines['right'].set_position(('axes', 1.10))
    ax_top_r2.patch.set_visible(False)
    l3, = ax_top_r2.plot(idx, bid_px.reindex(idx).values, label='Market1 Bid (px)',
                         color=COLOR_BID, linewidth=1.2, zorder=1)
    ax_top_r2.set_ylabel('Market1 Bid Price')

    title_str = (f"{symbol}  —  Spread / Funding / Bid"
                 f"   [{start_time.strftime('%Y-%m-%d %H:%M')}  →  {last_time.strftime('%Y-%m-%d %H:%M')}]")
    ax_top.set_title(title_str, fontsize=16, fontweight='bold', pad=12)

    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    leg = ax_top.legend(lines, labels, loc='upper left')
    leg.get_frame().set_alpha(0.9); leg.get_frame().set_linewidth(0.0)

    # 底部 2×2（顺序：FundingDiff | Rule-Median累计；Cumulative累计 | p40/p60累计）
    ax11 = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax12 = fig.add_subplot(gs[1, 1], sharex=ax_top)
    ax21 = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax22 = fig.add_subplot(gs[2, 1], sharex=ax_top)

    for ax in (ax11, ax12, ax21, ax22):
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax11.plot(series_fd.index, series_fd.values, color=C1, linewidth=1.6)
    ax11.set_title(f'{symbol}: OKX - Binance Funding Diff (bps)', fontsize=12)
    ax11.set_ylabel('bps')

    ax12.plot(series_adj.index, series_adj.values, color=C3, linewidth=1.6)
    ax12.set_title(f'{symbol}: Rule-Median Sign · Cumulative (bps)', fontsize=12)
    ax12.set_ylabel('bps')

    ax21.plot(series_cum.index, series_cum.values, color=C2, linewidth=1.8)
    ax21.set_title(f'{symbol}: Cumulative Funding Diff (bps)', fontsize=12)
    ax21.set_ylabel('bps')

    ax22.plot(series_sig.index, series_sig.values, color=C4, linewidth=1.6)
    ax22.set_title(f'{symbol}: Rule p40/p60 · Cumulative (bps)', fontsize=12)
    ax22.set_ylabel('bps')

    ax21.set_xlabel('FundingTime (Asia/Shanghai)')
    ax22.set_xlabel('FundingTime (Asia/Shanghai)')

    return fig

def generate_cf_report(symbol: str, st, et, last_time, lookback_window=3, mode='BN-OKX'):
    """
    一键生成 Cross-Exchange 报告：
      - 读取&处理价差数据（resample=60s, rolling=10）
      - 读取&处理资金费率（基础、Type1、Type2）
      - 生成大图（顶部三轴 + 底部2×2）
    
    Parameters:
    -----------
    symbol : str, 交易对符号
    st : datetime, 开始时间
    et : datetime, 结束时间
    last_time : datetime, 最后时间点
    lookback_window : int, 回看天数，默认3天
    mode : str, 交易所模式，如 'BN-OKX', 'BN-BYBIT', 'BN-GATE' 等
    
    返回：
      fig, results_dict, merged_df2, cf_depth_resampled
    """
    # 1) 价差
    cf_depth_resampled, spread_bid, spread_ask = load_spread_data_history_data(symbol, st, et, mode=mode)

    # 2) 资金费率（含 Type1 / Type2）
    funding_diff, merged_df, merged_df2 = load_funding_diffs(
        symbol, last_time, spread_bid, spread_ask, lookback_window, mode=mode
    )
    merged_df = merged_df.dropna()
    merged_df2 = merged_df2.dropna()
    # 3) 画图
    bid_px = cf_depth_resampled['bid_price_1_ex0']
    fig = plot_cf_panels(symbol, start_time=last_time - pd.Timedelta(days=lookback_window) + pd.Timedelta(minutes=10),
                         last_time=last_time,
                         spread_bid=spread_bid, bid_px=bid_px,
                         funding_diff=funding_diff,
                         merged_df=merged_df, merged_df2=merged_df2)

    results = {
        "symbol": symbol,
        "funding_diff_signed_cumsum": merged_df2['funding_diff_signed_cumsum'].iloc[-1],
        "funding_diff_adj_cumsum": merged_df2['funding_diff_adj_cumsum'].iloc[-1],
    }
    return fig,results,merged_df2,cf_depth_resampled




# PNL 分析相关

def load_symbol_pnl_dict(pkl_path):
    """
    加载保存的symbol_pnl_dict
    """
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
def filter_series(s, start_dt, end_dt, include_prev=True):
    if s is None or len(s) == 0:
        return s
    # 保证索引是 DatetimeIndex，且排序
    idx = pd.to_datetime(s.index)
    s2 = s.copy()
    s2.index = idx
    s2 = s2.sort_index()

    # 时区对齐（若一边有tz一边没有，这里做个简单处理）
    if getattr(start_dt, 'tzinfo', None) and s2.index.tz is None:
        start_dt = start_dt.tz_convert(None) if hasattr(start_dt, 'tz_convert') else start_dt.tz_localize(None)
        end_dt = end_dt.tz_convert(None) if hasattr(end_dt, 'tz_convert') else end_dt.tz_localize(None)
    if s2.index.tz and getattr(start_dt, 'tzinfo', None) is None:
        start_dt = start_dt.tz_localize(s2.index.tz)
        end_dt = end_dt.tz_localize(s2.index.tz)

    mask = (s2.index >= start_dt) & (s2.index <= end_dt)

    if include_prev:
        i = s2.index.searchsorted(start_dt, side='left') - 1  # 严格小于 start_dt 的最后一条
        if i >= 0:
            # 把这一条也保留
            mask[i] = True

    return s2.loc[mask]

def normalize_series(s):
    if s is None or s.empty:
        return s
    s_norm = s - s.iloc[0]
    return s_norm.iloc[1:]


def plot_symbol_and_portfolio_in_period(
    symbol_pnl_dict, 
    symbol, 
    start_date, 
    end_date, 
    initial_capital=None, 
    show=True, 
    type='pmpro',  # 新增type参数，默认pmpro
    show_portfolio=False,
    figure_type = 'plotly' #或matplotlib
):
    """
    用法示例：
        plot_symbol_and_portfolio_in_period(symbol_pnl_dict, 'SOON', '2025-07-08 04:05', '2025-07-09 04:05', type='pmpro')
    参数说明：
        symbol_pnl_dict: 你的收益数据字典，通常用 load_symbol_pnl_dict 加载
        symbol: 你要画的币种，比如 'SOON'
        start_date, end_date: 字符串，格式如 '2024-06-01 12:34'，精确到分钟
        initial_capital: 初始资金，默认 100000（仅对portfolio有效，单symbol分母见type说明）
        show: 是否直接在notebook里显示图，默认True
        type: 'pmpro' 或 'dcpro'，影响分母和portfolio初始资金
    返回：
        (symbol_fig, portfolio_fig) 两个plotly图对象
    """
    import pandas as pd
    from pandas import to_datetime


    # 解析时间，强制精确到分钟
    def parse_to_minute(dt_str):
        dt = to_datetime(dt_str)
        return dt.replace(second=0, microsecond=0)
    start_dt = parse_to_minute(start_date)
    end_dt = parse_to_minute(end_date)
    portfolio_initial_capital = C.PORTFOLIO_CONFIG[type]['total_capital']
    
    if initial_capital is None:
        denominator_ratio = C.PORTFOLIO_CONFIG[type]['denominator_ratio']
        symbol_denominator = portfolio_initial_capital*denominator_ratio

    else:
        # 如果指定了 initial_capital （比如是从仓位分配里读取的值），那算收益率和换手率时分母就用这个给定的值
        symbol_denominator = initial_capital  

    # 处理symbol
    symbol_data = symbol_pnl_dict.get(symbol, None)
    if symbol_data is None:
        print(f"Symbol {symbol} not found in symbol_pnl_dict")
        symbol_fig = None
    else:
        cum_pnl_combined = filter_series(symbol_data.get('cum_pnl_combined', pd.Series(dtype=float)),start_dt,end_dt)
        funding_pnl_series = filter_series(symbol_data.get('funding_pnl_series', pd.Series(dtype=float)),start_dt,end_dt)
        trade_pnl = filter_series(symbol_data.get('trade_pnl', pd.Series(dtype=float)),start_dt,end_dt)

        # 新增：归一化
        cum_pnl_combined_norm = normalize_series(cum_pnl_combined)
        funding_pnl_series_norm = normalize_series(funding_pnl_series)
        trade_pnl_norm = normalize_series(trade_pnl)

        if not cum_pnl_combined_norm.empty:
            total_ret = cum_pnl_combined_norm.iloc[-1]
            total_days = (cum_pnl_combined_norm.index[-1] - cum_pnl_combined_norm.index[0]).total_seconds() / 86400
            total_ret_rate = 365 * total_ret / symbol_denominator / total_days if total_days > 0 else 0
            print(total_ret, total_days, total_ret_rate)
        else:
            total_ret_rate = 0

        if not trade_pnl_norm.empty:
            spread_ret = trade_pnl_norm.iloc[-1]
            spread_ret_rate = 365 * spread_ret / symbol_denominator / total_days if total_days > 0 else 0
        else:
            spread_ret_rate = 0

        if not funding_pnl_series_norm.empty:
            funding_ret = funding_pnl_series_norm.iloc[-1] 
            funding_ret_rate = 365 * funding_ret / symbol_denominator / total_days if total_days > 0 else 0
        else:
            funding_ret_rate = 0

        turnover_rate = symbol_data.get('turnover_rate', 0)
        
        if figure_type == 'plotly':
            fig = go.Figure()
            if not cum_pnl_combined_norm.empty:
                fig.add_trace(go.Scatter(
                    x=cum_pnl_combined_norm.index, y=cum_pnl_combined_norm.values,
                    mode='lines+markers',
                    name='Cumulative PnL',
                    line=dict(color='royalblue'),
                    fill='tozeroy',
                    fillcolor='rgba(65,105,225,0.1)'
                ))
            if not funding_pnl_series_norm.empty:
                fig.add_trace(go.Scatter(
                    x=funding_pnl_series_norm.index, y=funding_pnl_series_norm.values,
                    mode='lines+markers',
                    name='Funding PnL',
                    line=dict(dash='dash', color='indianred')
                ))
            if not trade_pnl_norm.empty:
                fig.add_trace(go.Scatter(
                    x=trade_pnl_norm.index, y=trade_pnl_norm.values,
                    mode='lines+markers',
                    name='Trade PnL',
                    line=dict(color='deeppink')
                ))
            if not cum_pnl_combined_norm.empty:
                start_str = cum_pnl_combined_norm.index[0].strftime('%Y-%m-%d %H:%M')
                end_str = cum_pnl_combined_norm.index[-1].strftime('%Y-%m-%d %H:%M')
            else:
                start_str = parse_to_minute(start_date).strftime('%Y-%m-%d %H:%M')
                end_str = parse_to_minute(end_date).strftime('%Y-%m-%d %H:%M')
            fig.update_layout(
                title=f"{type} {symbol} PnL from {start_str} to {end_str}, Annualized Return: {total_ret_rate:.2%} | Spread: {spread_ret_rate:.2%} | Funding: {funding_ret_rate:.2%} | Turnover: {turnover_rate:.2f}",
                xaxis_title='Time (Asia/Shanghai)',
                yaxis_title='PnL (normalized to 0 at start)',
                legend=dict(font=dict(size=12)),
                template='plotly_white',
                height=700,
                width=1400
            )
            symbol_fig = fig
            if show:
                display(fig)
        elif figure_type == 'matplotlib':
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制累计PNL
            if not cum_pnl_combined_norm.empty:
                ax.plot(cum_pnl_combined_norm.index, cum_pnl_combined_norm.values, 
                        label='Cumulative PnL', color='royalblue', linewidth=2)
                ax.fill_between(cum_pnl_combined_norm.index, cum_pnl_combined_norm.values, 
                                alpha=0.1, color='royalblue')
            
            # 绘制资金费率PNL
            if not funding_pnl_series_norm.empty:
                ax.plot(funding_pnl_series_norm.index, funding_pnl_series_norm.values,
                        label='Funding PnL', color='indianred', linewidth=1.5, linestyle='--')
            
            # 绘制交易PNL
            if not trade_pnl_norm.empty:
                ax.plot(trade_pnl_norm.index, trade_pnl_norm.values,
                        label='Spread PnL', color='deeppink', linewidth=1.5)
            
            # 格式化
            start_str = start_dt.strftime('%Y-%m-%d %H:%M')
            end_str = end_dt.strftime('%Y-%m-%d %H:%M')
            
            title = (f"{symbol} PnL ({start_str} to {end_str})\n"
                    f"Total: {total_ret_rate:.2%} | Spread: {spread_ret_rate:.2%} | "
                    f"Funding: {funding_ret_rate:.2%}")
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('PnL (normalized)', fontsize=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            plt.tight_layout()
            symbol_fig = fig




    # 处理portfolio
    portfolio_data = symbol_pnl_dict.get('portfolio', None)
    if portfolio_data is None:
        print("Portfolio data not found in symbol_pnl_dict")
        portfolio_fig = None
    else:
        portfolio_cum_pnl = filter_series(portfolio_data.get('cum_pnl_combined', pd.Series(dtype=float)),start_dt,end_dt)
        portfolio_trade_pnl = filter_series(portfolio_data.get('trade_pnl', pd.Series(dtype=float)),start_dt,end_dt)
        portfolio_funding_pnl = filter_series(portfolio_data.get('funding_pnl_series', pd.Series(dtype=float)),start_dt,end_dt)
        portfolio_turnover_rate = portfolio_data.get('turnover_rate', 0)

        # 新增：归一化
        portfolio_cum_pnl_norm = normalize_series(portfolio_cum_pnl)
        portfolio_trade_pnl_norm = normalize_series(portfolio_trade_pnl)
        portfolio_funding_pnl_norm = normalize_series(portfolio_funding_pnl)

        # portfolio的初始资金由type决定
        portfolio_capital = portfolio_initial_capital

        if not portfolio_cum_pnl.empty:
            total_ret = portfolio_cum_pnl.iloc[-1] - portfolio_cum_pnl.iloc[0]
            total_days = (portfolio_cum_pnl.index[-1] - portfolio_cum_pnl.index[0]).total_seconds() / 86400
            total_ret_rate = 365 * total_ret / portfolio_capital / total_days if total_days > 0 else 0
        else:
            total_ret_rate = 0

        if not portfolio_trade_pnl.empty:
            spread_ret = portfolio_trade_pnl.iloc[-1] - portfolio_trade_pnl.iloc[0]
            spread_ret_rate = 365 * spread_ret / portfolio_capital / total_days if total_days > 0 else 0
        else:
            spread_ret_rate = 0

        if not portfolio_funding_pnl.empty:
            funding_ret = portfolio_funding_pnl.iloc[-1] - portfolio_funding_pnl.iloc[0]
            funding_ret_rate = 365 * funding_ret / portfolio_capital / total_days if total_days > 0 else 0
        else:
            funding_ret_rate = 0
        if figure_type == 'plotly':
            fig = go.Figure()
            if not portfolio_cum_pnl_norm.empty:
                fig.add_trace(go.Scatter(
                    x=portfolio_cum_pnl_norm.index, y=portfolio_cum_pnl_norm.values,
                    mode='lines',
                    name='Portfolio Total PnL',
                    line=dict(color='royalblue'),
                    fill='tozeroy',
                    fillcolor='rgba(65,105,225,0.1)'
                ))
            if not portfolio_trade_pnl_norm.empty:
                fig.add_trace(go.Scatter(
                    x=portfolio_trade_pnl_norm.index, y=portfolio_trade_pnl_norm.values,
                    mode='lines',
                    name='Portfolio Trade PnL',
                    line=dict(color='deeppink')
                ))
            if not portfolio_funding_pnl_norm.empty:
                fig.add_trace(go.Scatter(
                    x=portfolio_funding_pnl_norm.index, y=portfolio_funding_pnl_norm.values,
                    mode='lines',
                    name='Portfolio Funding PnL',
                    line=dict(dash='dash', color='indianred')
                ))
            if not portfolio_cum_pnl.empty:
                start_str = portfolio_cum_pnl.index[0].strftime('%Y-%m-%d %H:%M')
                end_str = portfolio_cum_pnl.index[-1].strftime('%Y-%m-%d %H:%M')
            else:
                start_str = parse_to_minute(start_date).strftime('%Y-%m-%d %H:%M')
                end_str = parse_to_minute(end_date).strftime('%Y-%m-%d %H:%M')
            fig.update_layout(
                title=(
                    f"{type} PnL from {start_str} to {end_str}  \n"
                    f"Annualized Return: {total_ret_rate:.2%} | Spread: {spread_ret_rate:.2%} | Funding: {funding_ret_rate:.2%} | Turnover: {portfolio_turnover_rate:.2f}"
                ),
                xaxis_title='Time',
                yaxis_title='PnL (normalized to 0 at start)',
                template='plotly_white',
                legend=dict(font=dict(size=12)),
                height=700,
                width=1400
            )
            portfolio_fig = fig
            if show_portfolio:
                display(fig)
        elif figure_type == 'matplotlib':
            # ---- matplotlib portfolio plot ----
            fig, ax = plt.subplots(figsize=(10, 6))

            # Total PnL
            if not portfolio_cum_pnl_norm.empty:
                ax.plot(
                    portfolio_cum_pnl_norm.index, portfolio_cum_pnl_norm.values,
                    label='Portfolio Total PnL', color='royalblue', linewidth=2
                )
                ax.fill_between(
                    portfolio_cum_pnl_norm.index, portfolio_cum_pnl_norm.values,
                    alpha=0.1, color='royalblue'
                )

            # Trade PnL
            if not portfolio_trade_pnl_norm.empty:
                ax.plot(
                    portfolio_trade_pnl_norm.index, portfolio_trade_pnl_norm.values,
                    label='Portfolio Trade PnL', color='deeppink', linewidth=1.5
                )

            # Funding PnL
            if not portfolio_funding_pnl_norm.empty:
                ax.plot(
                    portfolio_funding_pnl_norm.index, portfolio_funding_pnl_norm.values,
                    label='Portfolio Funding PnL', color='indianred', linewidth=1.5, linestyle='--'
                )

            # start/end string
            if not portfolio_cum_pnl.empty:
                start_str = portfolio_cum_pnl.index[0].strftime('%Y-%m-%d %H:%M')
                end_str = portfolio_cum_pnl.index[-1].strftime('%Y-%m-%d %H:%M')
            else:
                start_str = parse_to_minute(start_date).strftime('%Y-%m-%d %H:%M')
                end_str = parse_to_minute(end_date).strftime('%Y-%m-%d %H:%M')

            title = (
                f"{type} PnL from {start_str} to {end_str}\n"
                f"Annualized Return: {total_ret_rate:.2%} | Spread: {spread_ret_rate:.2%} | "
                f"Funding: {funding_ret_rate:.2%} | Turnover: {portfolio_turnover_rate:.2f}"
            )

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('PnL (normalized to 0 at start)')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

            plt.tight_layout()

            portfolio_fig = fig
            if show_portfolio:
                plt.show()     
            else:
                plt.close(fig)



    return symbol_fig

def plot_symbol_pnl_matplotlib(symbol, symbol_data, start_dt, end_dt, initial_capital):
    """
    使用matplotlib绘制单个symbol的PNL图表
    
    参数:
        symbol: 币种名称
        symbol_data: 该币种的PNL数据字典
        start_dt: 开始时间
        end_dt: 结束时间
        initial_capital: 初始资金 (2 * pos_limit)
    
    返回:
        fig: matplotlib图表对象
    """
    # 提取数据
    cum_pnl_combined = filter_series(
        symbol_data.get('cum_pnl_combined', pd.Series(dtype=float)), 
        start_dt, end_dt
    )
    funding_pnl_series = filter_series(
        symbol_data.get('funding_pnl_series', pd.Series(dtype=float)), 
        start_dt, end_dt
    )
    trade_pnl = filter_series(
        symbol_data.get('trade_pnl', pd.Series(dtype=float)), 
        start_dt, end_dt
    )
    
    # 归一化
    cum_pnl_combined_norm = normalize_series(cum_pnl_combined)
    funding_pnl_series_norm = normalize_series(funding_pnl_series)
    trade_pnl_norm = normalize_series(trade_pnl)
    
    # 计算年化收益率
    if not cum_pnl_combined_norm.empty:
        total_ret = cum_pnl_combined_norm.iloc[-1]
        total_days = (cum_pnl_combined_norm.index[-1] - cum_pnl_combined_norm.index[0]).total_seconds() / 86400
        total_ret_rate = 365 * total_ret / initial_capital / total_days if total_days > 0 else 0
    else:
        total_ret_rate = 0
        total_days = 0
    
    if not trade_pnl_norm.empty:
        spread_ret = trade_pnl_norm.iloc[-1]
        spread_ret_rate = 365 * spread_ret / initial_capital / total_days if total_days > 0 else 0
    else:
        spread_ret_rate = 0
    
    if not funding_pnl_series_norm.empty:
        funding_ret = funding_pnl_series_norm.iloc[-1] 
        funding_ret_rate = 365 * funding_ret / initial_capital / total_days if total_days > 0 else 0
    else:
        funding_ret_rate = 0
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制累计PNL
    if not cum_pnl_combined_norm.empty:
        ax.plot(cum_pnl_combined_norm.index, cum_pnl_combined_norm.values, 
                label='Cumulative PnL', color='royalblue', linewidth=2)
        ax.fill_between(cum_pnl_combined_norm.index, cum_pnl_combined_norm.values, 
                        alpha=0.1, color='royalblue')
    
    # 绘制资金费率PNL
    if not funding_pnl_series_norm.empty:
        ax.plot(funding_pnl_series_norm.index, funding_pnl_series_norm.values,
                label='Funding PnL', color='indianred', linewidth=1.5, linestyle='--')
    
    # 绘制交易PNL
    if not trade_pnl_norm.empty:
        ax.plot(trade_pnl_norm.index, trade_pnl_norm.values,
                label='Spread PnL', color='deeppink', linewidth=1.5)
    
    # 格式化
    start_str = start_dt.strftime('%Y-%m-%d %H:%M')
    end_str = end_dt.strftime('%Y-%m-%d %H:%M')
    
    title = (f"{symbol} PnL ({start_str} to {end_str})\n"
             f"Total: {total_ret_rate:.2%} | Spread: {spread_ret_rate:.2%} | "
             f"Funding: {funding_ret_rate:.2%}")
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('PnL (normalized)', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    return fig

def plot_top_bottom_symbols_in_period(
    symbol_pnl_dict, 
    start_date, 
    end_date, 
    n=5, 
    initial_capital=100000, 
    show=True, 
    type='pmpro'
):
    """
    画出给定区间内收益率最高和最低的n个symbol的表现（每个symbol单独画一张归一化收益曲线图）
    参数说明：
        symbol_pnl_dict: 你的收益数据字典
        start_date, end_date: 字符串，格式如 '2024-06-01 12:34'
        n: top/bottom的数量
        initial_capital, show, type: 同plot_symbol_and_portfolio_in_period
    返回：
        (top_figs, bottom_figs) 两个list，分别为top n和bottom n symbol的plotly图对象
    """

    def parse_to_minute(dt_str):
        dt = to_datetime(dt_str)
        return dt.replace(second=0, microsecond=0)
    start_dt = parse_to_minute(start_date)
    end_dt = parse_to_minute(end_date)


    # 根据type设置分母
    if type == 'pmpro':
        symbol_denominator = 4000
    elif type == 'dcpro':
        symbol_denominator = 40000
    elif type == 'dcpro5':
        symbol_denominator = 21000
    elif type == 'dcpro3':
        symbol_denominator = 200000
    elif type == 'dcpro6':
        symbol_denominator = 80000
    elif type == 'pmtest4':
        symbol_denominator = 200
    else:
        symbol_denominator = initial_capital  # fallback

    # 计算所有symbol的收益率
    symbol_returns = []
    for symbol, data in symbol_pnl_dict.items():
        if symbol == 'portfolio':
            continue
        cum_pnl = filter_series(data.get('cum_pnl_combined', pd.Series(dtype=float)),start_dt,end_dt)
        if cum_pnl is None or cum_pnl.empty:
            continue
        # 只考虑区间内有数据的symbol
        if len(cum_pnl) < 2:
            continue
        ret = (cum_pnl.iloc[-1] - cum_pnl.iloc[0]) / symbol_denominator
        symbol_returns.append((symbol, ret, cum_pnl))

    if not symbol_returns:
        print("No symbol has valid PnL data in the given period.")
        return None, None

    # 排序
    symbol_returns_sorted = sorted(symbol_returns, key=lambda x: x[1], reverse=True)
    top_symbols = symbol_returns_sorted[:n]
    bottom_symbols = symbol_returns_sorted[-n:]

    # 画top n，每个symbol单独画
    top_figs = []
    for symbol, ret, cum_pnl in top_symbols:
        norm_curve = normalize_series(cum_pnl)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=norm_curve.index, y=norm_curve.values,
            mode='lines',
            name=f"{symbol} ({ret:.2%})"
        ))
        fig.update_layout(
            title=f"Top Symbol: {symbol} ({ret:.2%})\nfrom {start_dt} to {end_dt}",
            xaxis_title='Time',
            yaxis_title='PnL (normalized to 0 at start)',
            template='plotly_white',
            legend=dict(font=dict(size=12)),
            height=700,
            width=1400
        )
        if show:
            display(fig)
        top_figs.append(fig)

    # 画bottom n，每个symbol单独画
    bottom_figs = []
    for symbol, ret, cum_pnl in bottom_symbols:
        norm_curve = normalize_series(cum_pnl)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=norm_curve.index, y=norm_curve.values,
            mode='lines',
            name=f"{symbol} ({ret:.2%})"
        ))
        fig.update_layout(
            title=f"Bottom Symbol: {symbol} ({ret:.2%})\nfrom {start_dt} to {end_dt}",
            xaxis_title='Time',
            yaxis_title='PnL (normalized to 0 at start)',
            template='plotly_white',
            legend=dict(font=dict(size=12)),
            height=700,
            width=1400
        )
        if show:
            display(fig)
        bottom_figs.append(fig)


# 用法举例（在notebook里直接运行下面这行即可画图）：
# plot_top_bottom_symbols_in_period(symbol_pnl_dict, '2025-07-08 04:05', '2025-07-09 04:05', n=5, type='pmpro')
