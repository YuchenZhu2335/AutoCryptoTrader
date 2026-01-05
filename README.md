# AutoCryptoTrader

基于 GitHub Actions 的免费云端模拟交易系统，每天自动运行 V5 策略（MA200过滤 + 唐奇安通道 + ATR止损）。

## 策略说明

### V5 策略核心逻辑

1. **MA200 趋势过滤**: 只在长期趋势向上时开仓（收盘价 > MA200）
2. **唐奇安通道突破**: 收盘价突破20日唐奇安上轨时买入
3. **ATR 追踪止损**: 使用2倍ATR作为动态止损线，跟随价格上涨

### 策略参数

- 初始资金: 100,000 USDT
- 交易费率: 0.1%（双边收取）
- 唐奇安周期: 20日
- ATR倍数: 2.0
- MA200周期: 200日

## 文件结构

```
AutoCryptoTrader/
├── simulation_trader.py      # 核心交易逻辑
├── requirements.txt          # Python依赖
├── .github/workflows/        # GitHub Actions配置
│   └── daily_run.yml
└── data/                     # 交易记录（自动生成）
    ├── portfolio_state.json  # 投资组合状态
    ├── trade_history.csv    # 交易历史记录
    └── daily_balance.csv    # 每日权益记录
```

## 自动化执行

系统每天 UTC 0:05（北京时间 8:05）自动运行一次，执行以下操作：

1. 加载投资组合状态
2. 获取最新市场数据
3. 执行策略逻辑判断
4. 如有交易信号，执行买卖操作
5. 记录交易和每日权益
6. 自动提交数据到仓库

## 复盘文档

### 交易历史 (`data/trade_history.csv`)

记录所有买卖操作，包含：
- 时间戳
- 操作类型（BUY/SELL）
- 成交价格
- 交易原因
- 持仓数量
- 现金余额
- 总权益

### 每日权益 (`data/daily_balance.csv`)

记录每日投资组合权益，包含：
- 日期
- 现金余额
- 持仓数量
- 当前价格
- 总权益

用于绘制资金曲线和计算策略表现。

## 手动操作说明

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行一次交易检查
python simulation_trader.py
```

### 查看交易记录

```bash
# 查看交易历史
cat data/trade_history.csv

# 查看每日权益
cat data/daily_balance.csv

# 查看当前状态
cat data/portfolio_state.json
```

### 手动触发 GitHub Actions

1. 访问仓库的 Actions 页面
2. 选择 "Daily Trading Simulation" 工作流
3. 点击 "Run workflow" 按钮

## 注意事项

- 所有交易均为**模拟交易**，不涉及真实资金
- 策略基于历史数据回测，不保证未来表现
- 数据文件会自动提交到仓库，请勿手动修改
- 如需重置策略，删除 `data/portfolio_state.json` 即可

## 技术栈

- Python 3.12+
- ccxt: 加密货币数据获取
- pandas: 数据处理
- numpy: 数值计算
- GitHub Actions: 自动化执行

---

**最后更新**: 2025-01-04
