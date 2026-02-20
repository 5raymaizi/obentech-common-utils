# 项目配置

## 目录结构

### 数据目录
主要数据位置: `/Volumes/T7/Obentech/`

- **scored_df/** - 按交易所命名，包含每天计算的特征
- **leverage_and_MMR/** - 按交易所命名，包含阶梯杠杆数据
- **historyDepthData/** - 所有交易所的分钟级别历史价差数据
- **fundingRateData/** - 按交易所命名，历史资金费率数据

### 代码目录
- 主要代码位置: `/Users/rayxu/Documents/Obentech_code`
- 里面有两个项目文件夹（AM，CF）和一个工具函数文件夹（COMMON_UTILS/src/common_utils下面)
- 语言: Python

## 项目说明
数字货币的跨所套利和单所套利项目


## 主要代码模块
- **服务器自动化脚本**:s512_script_v2.py, s517_script.py (跨所), s517_script_single_exchange.py (单所)
- **回测系统**: CF的是backtesting.py, CF跨所回测V10086.ipynb; AM的是 hedge_backtest这个folder下面
- **数据获取**: dcdl_*.py系列
- **配置文件**: CONFIG.py
- **工具函数**: utils_Sep.py
- **自动化相关**: 周报.ipynb(自动生成周报), CF脚本_pos_0.ipynb(调整实盘杠杆)


## 工作流程
- **下载历史数据**: 各项目文件夹下的 `dcdl_collect_data.py` 可下载指定日期范围的 depth/trade 数据（从 dcdl.digifinex.org），下载后自动转为 parquet 格式，存到 `/Volumes/T7/Obentech/data/` 下
