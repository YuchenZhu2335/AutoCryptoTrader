"""
云端模拟交易系统 - 基于 V5 策略
这是一个单次运行的脚本（One-shot script），用于每天被 GitHub Actions 定时唤醒。

策略逻辑：MA200过滤 + 唐奇安通道 + ATR止损
数据获取：优先使用 ccxt.binance，失败则切换到 yfinance

状态管理：
- 读取 data/portfolio_state.json（如果不存在则初始化 10万U 资金）
- 将最新的持仓、止损位、总资产保存回 data/portfolio_state.json

交易记录：
- 买卖操作追加写入 data/trade_history.csv
- 每日权益追加写入 data/daily_balance.csv（用于画资金曲线）
"""

import ccxt
import pandas as pd
import numpy as np
import json
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ========== 配置参数 ==========
INITIAL_CAPITAL = 100000  # 初始资金 10万 USDT
FEE_RATE = 0.001  # 交易费率 0.1%（双边收取）
ATR_PERIOD = 14  # ATR 计算周期
MA200_PERIOD = 200  # MA200 周期

# V5 集成策略配置（4个子策略，每个权重25%）
STRATEGY_CONFIGS = [
    {'donchian_period': 20, 'atr_multiplier': 5.0, 'weight': 0.25},
    {'donchian_period': 30, 'atr_multiplier': 4.5, 'weight': 0.25},
    {'donchian_period': 40, 'atr_multiplier': 4.0, 'weight': 0.25},
    {'donchian_period': 50, 'atr_multiplier': 3.5, 'weight': 0.25},
]

# 数据文件路径
DATA_DIR = Path(__file__).parent / "data"
PORTFOLIO_STATE_FILE = DATA_DIR / "portfolio_state.json"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history.csv"
DAILY_BALANCE_FILE = DATA_DIR / "daily_balance.csv"

# 确保数据目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ========== 数据获取函数 ==========
def fetch_data_ccxt(symbol='BTC/USDT', days=730):
    """使用 ccxt 从币安获取日线数据"""
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 获取日线数据（1d）
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=int(start_time.timestamp() * 1000))
        
        # 转换为 DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"Successfully fetched {len(df)} days of {symbol} data from Binance")
        return df
        
    except Exception as e:
        print(f"Failed to fetch data from Binance: {e}")
        return None


def fetch_data_yfinance(symbol='BTC/USDT', days=730):
    """使用 yfinance 获取 BTC 数据（作为备选方案）
    
    注意：yfinance 使用 BTC-USD 作为交易对符号
    """
    try:
        # yfinance 使用 BTC-USD 作为交易对
        ticker_symbol = 'BTC-USD'
        
        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 获取日线数据
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"No data returned from Yahoo Finance for {ticker_symbol}")
            return None
        
        # 重命名列以匹配标准格式（yfinance 返回的列名可能是大写）
        df.columns = [col.lower() for col in df.columns]
        
        # 确保包含所需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in Yahoo Finance data")
            return None
        
        # 选择所需的列
        df = df[required_columns]
        
        # 确保索引是 datetime 类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        print(f"Successfully fetched {len(df)} days of {ticker_symbol} data from Yahoo Finance")
        return df
        
    except Exception as e:
        print(f"Failed to fetch data from Yahoo Finance: {e}")
        return None


def get_data(symbol='BTC/USDT', days=730):
    """获取数据，优先使用 ccxt，失败则使用 yfinance
    
    参数:
        symbol: 交易对符号（如 'BTC/USDT'）
        days: 需要获取的天数
    
    返回:
        DataFrame: 包含 OHLCV 数据的 DataFrame，失败则返回 None
    """
    # 优先尝试从币安获取数据
    print(f"Attempting to fetch {symbol} data from Binance...")
    df = fetch_data_ccxt(symbol=symbol, days=days)
    
    # 如果币安获取失败，切换到 Yahoo Finance
    if df is None or len(df) == 0:
        print("Binance data fetch failed, switching to Yahoo Finance...")
        df = fetch_data_yfinance(symbol=symbol, days=days)
    
    # 验证数据
    if df is None or len(df) == 0:
        print("ERROR: Failed to fetch data from both Binance and Yahoo Finance")
        return None
    
    if len(df) < MA200_PERIOD:
        print(f"WARNING: Data length ({len(df)}) is less than required MA200 period ({MA200_PERIOD})")
    
    return df


# ========== 策略计算函数 ==========
def calculate_donchian_upper(df, period=20):
    """计算唐奇安通道上轨（过去N天的最高价）"""
    return df['high'].rolling(window=period).max().shift(1)


def calculate_atr(df, period=ATR_PERIOD):
    """计算 ATR (Average True Range)"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_ma200(df):
    """计算200日简单移动平均线"""
    return df['close'].rolling(window=MA200_PERIOD).mean()


# ========== 状态管理函数 ==========
def load_portfolio_state():
    """加载投资组合状态"""
    if PORTFOLIO_STATE_FILE.exists():
        try:
            with open(PORTFOLIO_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            print(f"Portfolio state loaded: {state}")
            return state
        except Exception as e:
            print(f"Failed to load state file: {e}, using default values")
    
    # 初始化状态（支持多子策略）
    state = {
        'symbol': 'BTC/USDT',
        'in_position': False,
        'buy_price': None,
        'trailing_stop': None,
        'atr_at_buy': None,
        'cash': INITIAL_CAPITAL,
        'position_size': 0.0,  # 持仓数量（BTC）
        'total_equity': INITIAL_CAPITAL,  # 总权益
        'last_update': None,
        'sub_strategies': {}  # 存储每个子策略的状态
    }
    print(f"Initialized portfolio state: {state}")
    return state


def save_portfolio_state(state):
    """保存投资组合状态"""
    state['last_update'] = datetime.now().isoformat()
    try:
        with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"Portfolio state saved successfully")
    except Exception as e:
        print(f"Failed to save state file: {e}")


# ========== 交易记录函数 ==========
def log_trade(action, price, reason, state):
    """记录交易操作"""
    trade_record = {
        'timestamp': datetime.now().isoformat(),
        'action': action,  # 'BUY' 或 'SELL'
        'price': price,
        'reason': reason,
        'position_size': state['position_size'],
        'cash': state['cash'],
        'total_equity': state['total_equity']
    }
    
    # 追加到 CSV 文件
    df_trade = pd.DataFrame([trade_record])
    file_exists = TRADE_HISTORY_FILE.exists()
    df_trade.to_csv(TRADE_HISTORY_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
    print(f"Trade logged: {action} @ {price} USDT, reason: {reason}")


def log_daily_balance(state, current_price):
    """记录每日权益"""
    # 计算当前总权益
    if state['in_position']:
        total_equity = state['cash'] + state['position_size'] * current_price
    else:
        total_equity = state['cash']
    
    balance_record = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'cash': state['cash'],
        'position_size': state['position_size'],
        'current_price': current_price,
        'total_equity': total_equity
    }
    
    # 追加到 CSV 文件
    df_balance = pd.DataFrame([balance_record])
    file_exists = DAILY_BALANCE_FILE.exists()
    df_balance.to_csv(DAILY_BALANCE_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
    print(f"Daily balance logged: {total_equity:.2f} USDT (cash: {state['cash']:.2f}, position: {state['position_size']:.6f} BTC)")


# ========== 策略执行函数 ==========
def run_single_strategy_signal(df, donchian_period, atr_multiplier, state):
    """
    运行单个子策略，返回信号和状态
    
    返回:
        signal: 1 (买入), -1 (卖出), 0 (无信号)
        sub_state: 子策略的状态信息
    """
    latest = df.iloc[-1]
    current_close = latest['close']
    current_low = latest['low']
    
    # 计算该子策略的指标
    df['donchian_upper'] = calculate_donchian_upper(df, period=donchian_period)
    df['atr'] = calculate_atr(df, period=ATR_PERIOD)
    df['ma200'] = calculate_ma200(df)
    
    latest_donchian = df['donchian_upper'].iloc[-1]
    latest_atr = df['atr'].iloc[-1]
    latest_ma200 = df['ma200'].iloc[-1]
    
    # 获取或初始化子策略状态
    sub_key = f"{donchian_period}_{atr_multiplier}"
    if sub_key not in state.get('sub_strategies', {}):
        sub_state = {
            'in_position': False,
            'buy_price': None,
            'trailing_stop': None,
            'atr_at_buy': None
        }
    else:
        sub_state = state['sub_strategies'][sub_key].copy()
    
    signal = 0
    
    if not sub_state['in_position']:
        # 空仓状态：检查买入信号
        if (not np.isnan(latest_donchian) and 
            not np.isnan(latest_ma200) and
            current_close > latest_donchian and 
            current_close > latest_ma200):
            
            signal = 1
            sub_state['in_position'] = True
            sub_state['buy_price'] = float(current_close)
            sub_state['atr_at_buy'] = float(latest_atr if not np.isnan(latest_atr) else df['atr'].iloc[-2])
            sub_state['trailing_stop'] = float(sub_state['buy_price'] - atr_multiplier * sub_state['atr_at_buy'])
    else:
        # 持仓状态：更新止损线并检查卖出信号
        # 计算新的止损线（跟随价格上涨）
        new_stop = current_close - atr_multiplier * sub_state['atr_at_buy']
        if new_stop > sub_state['trailing_stop']:
            sub_state['trailing_stop'] = float(new_stop)
        
        # 检查卖出信号：价格触及止损线
        if current_low <= sub_state['trailing_stop']:
            signal = -1
            sub_state['in_position'] = False
            sub_state['buy_price'] = None
            sub_state['trailing_stop'] = None
            sub_state['atr_at_buy'] = None
    
    return signal, sub_state


def run_strategy_logic(df, state):
    """
    执行 V5 集成策略逻辑（多子策略组合）
    
    返回:
        action: 'BUY', 'SELL', 或 None
        reason: 交易原因
        new_state: 更新后的状态
    """
    # 获取最新数据
    latest = df.iloc[-1]
    current_close = latest['close']
    current_low = latest['low']
    
    # 计算 MA200（所有子策略共用）
    df['ma200'] = calculate_ma200(df)
    latest_ma200 = df['ma200'].iloc[-1]
    
    # 为每个子策略计算信号
    sub_signals = {}
    new_sub_states = {}
    ensemble_position = 0.0  # 集成持仓比例（0-1）
    
    for config in STRATEGY_CONFIGS:
        donchian_period = config['donchian_period']
        atr_multiplier = config['atr_multiplier']
        weight = config['weight']
        
        signal, sub_state = run_single_strategy_signal(
            df.copy(), donchian_period, atr_multiplier, state
        )
        
        sub_key = f"{donchian_period}_{atr_multiplier}"
        sub_signals[sub_key] = signal
        new_sub_states[sub_key] = sub_state
        
        # 计算集成持仓比例（加权平均）
        if sub_state['in_position']:
            ensemble_position += weight
    
    # 根据集成信号决定交易动作
    action = None
    reason = None
    new_state = state.copy()
    new_state['sub_strategies'] = new_sub_states
    
    # 判断买入：如果集成持仓比例从 < 0.5 变为 >= 0.5（多数子策略看多）
    # 判断卖出：如果集成持仓比例从 >= 0.5 变为 < 0.5（多数子策略看空）
    previous_position = 0.0
    if 'sub_strategies' in state:
        for config in STRATEGY_CONFIGS:
            sub_key = f"{config['donchian_period']}_{config['atr_multiplier']}"
            if sub_key in state['sub_strategies']:
                if state['sub_strategies'][sub_key].get('in_position', False):
                    previous_position += config['weight']
    
    if not state['in_position']:
        # 空仓状态：检查买入信号（集成持仓比例 >= 0.5，且之前 < 0.5）
        if ensemble_position >= 0.5 and previous_position < 0.5:
            # 执行买入
            buy_price = current_close
            
            # 使用加权平均的 ATR 和止损
            weighted_atr = 0.0
            weighted_stop = 0.0
            for config in STRATEGY_CONFIGS:
                sub_key = f"{config['donchian_period']}_{config['atr_multiplier']}"
                if new_sub_states[sub_key]['in_position']:
                    weighted_atr += config['weight'] * new_sub_states[sub_key]['atr_at_buy']
                    weighted_stop += config['weight'] * new_sub_states[sub_key]['trailing_stop']
            
            # 如果所有子策略都看多，使用加权平均；否则使用最保守的止损
            if weighted_stop > 0:
                initial_stop = weighted_stop
                atr_at_buy = weighted_atr
            else:
                # 使用最保守的止损（最高的止损线）
                stops = [s['trailing_stop'] for s in new_sub_states.values() if s.get('trailing_stop')]
                initial_stop = max(stops) if stops else buy_price * 0.9  # 默认10%止损
                atr_at_buy = df['atr'].iloc[-1] if not np.isnan(df['atr'].iloc[-1]) else df['atr'].iloc[-2]
            
            # 计算可买入数量（扣除手续费）
            available_cash = state['cash'] * (1 - FEE_RATE)
            position_size = available_cash / buy_price
            
            # 更新状态
            new_state['in_position'] = True
            new_state['buy_price'] = float(buy_price)
            new_state['trailing_stop'] = float(initial_stop)
            new_state['atr_at_buy'] = float(atr_at_buy)
            new_state['position_size'] = float(position_size)
            new_state['cash'] = 0.0
            
            action = 'BUY'
            active_strategies = [f"{c['donchian_period']}/{c['atr_multiplier']}" 
                                for c in STRATEGY_CONFIGS 
                                if new_sub_states[f"{c['donchian_period']}_{c['atr_multiplier']}"]['in_position']]
            reason = f"Ensemble signal: {len(active_strategies)}/4 strategies bullish, price > MA200 ({latest_ma200:.2f})"
    
    else:
        # 持仓状态：更新止损线并检查卖出信号
        buy_price = state['buy_price']
        atr_at_buy = state['atr_at_buy']
        current_stop = state['trailing_stop']
        
        # 计算新的止损线（使用所有持仓子策略中最保守的止损）
        active_stops = []
        for config in STRATEGY_CONFIGS:
            sub_key = f"{config['donchian_period']}_{config['atr_multiplier']}"
            if new_sub_states[sub_key]['in_position'] and new_sub_states[sub_key].get('trailing_stop'):
                active_stops.append(new_sub_states[sub_key]['trailing_stop'])
        
        if active_stops:
            # 使用最保守的止损（最高的止损线，即最不容易触发）
            new_stop = max(active_stops)
            if new_stop > current_stop:
                current_stop = new_stop
                new_state['trailing_stop'] = float(current_stop)
        
        # 检查卖出信号：集成持仓比例 < 0.5 或价格触及止损线
        if ensemble_position < 0.5 or current_low <= current_stop:
            # 执行卖出
            sell_price = current_stop if current_low <= current_stop else current_close
            
            # 计算卖出后现金（扣除手续费）
            position_size = state['position_size']
            cash_after_sell = position_size * sell_price * (1 - FEE_RATE)
            
            # 更新状态
            new_state['in_position'] = False
            new_state['buy_price'] = None
            new_state['trailing_stop'] = None
            new_state['atr_at_buy'] = None
            new_state['cash'] = float(cash_after_sell)
            new_state['position_size'] = 0.0
            
            action = 'SELL'
            if current_low <= current_stop:
                reason = f"Hit trailing stop ({current_stop:.2f})"
            else:
                reason = f"Ensemble signal: {sum(1 for s in new_sub_states.values() if s['in_position'])}/4 strategies bearish"
    
    # 计算总权益
    if new_state['in_position']:
        new_state['total_equity'] = new_state['cash'] + new_state['position_size'] * current_close
    else:
        new_state['total_equity'] = new_state['cash']
    
    return action, reason, new_state


# ========== 主函数 ==========
def main():
    """主函数：执行单次交易检查"""
    print("="*60)
    print(f"Cloud Trading System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. 加载投资组合状态
    state = load_portfolio_state()
    
    # 2. 获取最新数据（优先币安，失败则使用 Yahoo Finance）
    symbol = state.get('symbol', 'BTC/USDT')
    print(f"\nFetching {symbol} data...")
    df = get_data(symbol=symbol, days=730)
    
    if df is None or len(df) < MA200_PERIOD:
        print("ERROR: Insufficient data, cannot execute strategy")
        return
    
    # 3. 执行策略逻辑（V5 集成策略）
    print(f"\nExecuting V5 Ensemble Strategy...")
    print(f"Strategy pool: {len(STRATEGY_CONFIGS)} sub-strategies")
    for idx, config in enumerate(STRATEGY_CONFIGS):
        print(f"  Sub-strategy {idx+1}: Period={config['donchian_period']}, ATR={config['atr_multiplier']}, Weight={config['weight']*100:.0f}%")
    action, reason, new_state = run_strategy_logic(df, state)
    
    # 4. 如果有交易，记录交易
    if action:
        log_trade(action, 
                 df.iloc[-1]['close'], 
                 reason, 
                 new_state)
    
    # 5. 记录每日权益
    current_price = df.iloc[-1]['close']
    log_daily_balance(new_state, current_price)
    
    # 6. 保存状态
    save_portfolio_state(new_state)
    
    # 7. 打印摘要
    print("\n" + "="*60)
    print("Trading Summary:")
    print(f"  Position: {'In Position' if new_state['in_position'] else 'No Position'}")
    if new_state['in_position']:
        print(f"  Buy Price: {new_state['buy_price']:.2f} USDT")
        print(f"  Current Stop: {new_state['trailing_stop']:.2f} USDT")
        print(f"  Position Size: {new_state['position_size']:.6f} BTC")
    print(f"  Cash: {new_state['cash']:.2f} USDT")
    print(f"  Total Equity: {new_state['total_equity']:.2f} USDT")
    if action:
        print(f"  Action: {action} - {reason}")
    else:
        print(f"  Action: No trade")
    print("="*60)


if __name__ == "__main__":
    main()


