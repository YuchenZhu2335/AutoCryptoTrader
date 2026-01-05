"""
云端模拟交易系统 - 基于 V5 策略
这是一个单次运行的脚本（One-shot script），用于每天被 GitHub Actions 定时唤醒。

策略逻辑：MA200过滤 + 唐奇安通道 + ATR止损
数据获取：使用 ccxt.binance 获取 BTC/USDT 日线数据

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
from datetime import datetime, timedelta
from pathlib import Path

# ========== 配置参数 ==========
INITIAL_CAPITAL = 100000  # 初始资金 10万 USDT
FEE_RATE = 0.001  # 交易费率 0.1%（双边收取）
DONCHIAN_PERIOD = 20  # 唐奇安通道周期
ATR_MULTIPLIER = 2.0  # ATR 倍数
ATR_PERIOD = 14  # ATR 计算周期
MA200_PERIOD = 200  # MA200 周期

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


# ========== 策略计算函数 ==========
def calculate_donchian_upper(df, period=DONCHIAN_PERIOD):
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
    
    # 初始化状态
    state = {
        'symbol': 'BTC/USDT',
        'in_position': False,
        'buy_price': None,
        'trailing_stop': None,
        'atr_at_buy': None,
        'cash': INITIAL_CAPITAL,
        'position_size': 0.0,  # 持仓数量（BTC）
        'total_equity': INITIAL_CAPITAL,  # 总权益
        'last_update': None
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
def run_strategy_logic(df, state):
    """
    执行策略逻辑（基于 V5 策略）
    
    返回:
        action: 'BUY', 'SELL', 或 None
        reason: 交易原因
        new_state: 更新后的状态
    """
    # 获取最新数据（最后一行）
    latest = df.iloc[-1]
    current_close = latest['close']
    current_high = latest['high']
    current_low = latest['low']
    
    # 计算指标（使用足够的历史数据）
    df['donchian_upper'] = calculate_donchian_upper(df, period=DONCHIAN_PERIOD)
    df['atr'] = calculate_atr(df, period=ATR_PERIOD)
    df['ma200'] = calculate_ma200(df)
    
    latest_donchian = df['donchian_upper'].iloc[-1]
    latest_atr = df['atr'].iloc[-1]
    latest_ma200 = df['ma200'].iloc[-1]
    
    action = None
    reason = None
    new_state = state.copy()
    
    if not state['in_position']:
        # 空仓状态：检查买入信号
        # 条件：收盘价突破唐奇安上轨 且 收盘价 > MA200（趋势过滤）
        if (not np.isnan(latest_donchian) and 
            not np.isnan(latest_ma200) and
            current_close > latest_donchian and 
            current_close > latest_ma200):
            
            # 执行买入
            buy_price = current_close
            atr_at_buy = latest_atr if not np.isnan(latest_atr) else df['atr'].iloc[-2]
            initial_stop = buy_price - ATR_MULTIPLIER * atr_at_buy
            
            # 计算可买入数量（扣除手续费）
            available_cash = state['cash'] * (1 - FEE_RATE)  # 买入时扣除手续费
            position_size = available_cash / buy_price
            
            # 更新状态
            new_state['in_position'] = True
            new_state['buy_price'] = float(buy_price)
            new_state['trailing_stop'] = float(initial_stop)
            new_state['atr_at_buy'] = float(atr_at_buy)
            new_state['position_size'] = float(position_size)
            new_state['cash'] = 0.0
            
            action = 'BUY'
            reason = f"Breakout above Donchian upper ({latest_donchian:.2f}) and price > MA200 ({latest_ma200:.2f})"
    
    else:
        # 持仓状态：更新止损线并检查卖出信号
        buy_price = state['buy_price']
        atr_at_buy = state['atr_at_buy']
        current_stop = state['trailing_stop']
        
        # 计算新的止损线（跟随价格上涨）
        new_stop = current_close - ATR_MULTIPLIER * atr_at_buy
        if new_stop > current_stop:
            current_stop = new_stop
            new_state['trailing_stop'] = float(current_stop)
        
        # 检查卖出信号：价格触及止损线
        if current_low <= current_stop:
            # 执行卖出
            sell_price = current_stop  # 以止损价成交
            
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
            reason = f"Hit trailing stop ({current_stop:.2f})"
    
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
    
    # 2. 获取最新数据
    symbol = state.get('symbol', 'BTC/USDT')
    print(f"\nFetching {symbol} data...")
    df = fetch_data_ccxt(symbol=symbol, days=730)
    
    if df is None or len(df) < MA200_PERIOD:
        print("Insufficient data, cannot execute strategy")
        return
    
    # 3. 执行策略逻辑
    print(f"\nExecuting strategy logic...")
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

