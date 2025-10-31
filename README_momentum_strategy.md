# 短期动量策略启动器

## 概述

这是一个基于trades_window数据的短期动量策略启动器，实时分析BTC-FDUSD交易数据并生成动量交易信号。

## 快速开始

### 1. 启动策略

```bash
# 使用默认配置启动
venv/bin/python momentum_strategy.py

# 使用自定义配置启动
venv/bin/python momentum_strategy.py --config config/my_config.yaml

# 查看帮助信息
venv/bin/python momentum_strategy.py --help
```

### 2. 配置文件

编辑 `config/momentum_strategy.yaml` 来调整策略参数：

```yaml
# 分析器配置
analyzer:
  window_size_minutes: 5          # 分析时间窗口（分钟）
  min_trades: 10                  # 最小交易数量
  min_volume: 0.01               # 最小成交量
  buy_threshold: 0.15            # 买入阈值
  sell_threshold: -0.15          # 卖出阈值

# 数据源配置
data:
  symbol: "BTCFDUSD"             # 交易对
  data_source: "redis"           # 数据源: redis, mock, file, websocket

# Redis配置（仅在使用redis数据源时需要）
redis:
  host: "localhost"              # Redis服务器地址
  port: 6379                     # Redis端口
  db: 0                          # Redis数据库编号
  storage_dir: "storage"         # 数据存储目录

# 输出配置
output:
  console: true                  # 控制台输出
  file: true                     # 文件输出
  file_path: "logs/momentum_signals.log"
  signal_file: "signals/latest_signal.json"
```

### 3. 输出文件

- **日志文件**: `logs/momentum_strategy.log` - 策略运行日志
- **信号日志**: `logs/momentum_signals.log` - 交易信号记录（CSV格式）
- **最新信号**: `signals/latest_signal.json` - 最新的分析结果（JSON格式）

## 信号解读

### 信号方向
- 🟢 **BUY**: 买入信号，预期价格上涨
- 🔴 **SELL**: 卖出信号，预期价格下跌
- 🟡 **NEUTRAL**: 中性信号，无明显趋势

### 关键指标
- **强度 (0.0-1.0)**: 信号强度，越高表示信号越可靠
- **置信度 (0.0-1.0)**: 分析结果的可信程度
- **原始分数**: 综合动量分数，用于确定信号方向

### 动量指标
- **价格动量**: 基于价格变化的动量指标
- **成交量动量**: 基于成交量不平衡的动量指标
- **订单流动量**: 基于买卖压力的动量指标
- **波动率调整**: 经过波动率调整后的动量指标

## 使用示例

### 控制台输出示例

```
================================================================================
🚀 短期动量信号 #1
================================================================================
📊 交易对: BTCFDUSD
⏰ 分析时间: 2025-10-28 13:21:13
🎯 信号方向: 🟡 NEUTRAL
💪 信号强度: 0.212
🔍 置信度: 0.517
📈 原始分数: -0.1061
🏪 市场条件: ranging
📋 交易数量: 20
⚡ 处理时间: 1.05ms

📊 关键指标:
  💰 价格动量: +0.0398
  📊 成交量动量: -0.2325
  🔄 订单流动量: -0.2170
  📈 波动率调整: +0.0398
  ⚖️ 成交量不平衡: -0.5926
  📉 实现波动率: 0.0000
================================================================================
```

### 读取最新信号

```python
import json

# 读取最新信号
with open('signals/latest_signal.json', 'r') as f:
    signal_data = json.load(f)

# 获取信号信息
signal = signal_data['signal']
direction = signal['direction']
strength = signal['strength']
confidence = signal['confidence']

print(f"信号方向: {direction}")
print(f"信号强度: {strength:.3f}")
print(f"置信度: {confidence:.3f}")
```

## 测试

### 运行测试脚本

```bash
# 运行功能测试
venv/bin/python test_strategy_launcher.py

# 运行调试脚本
venv/bin/python debug_momentum_launcher.py
```

### 运行单元测试

```bash
# 运行所有测试
venv/bin/pytest tests/unit/test_short_term_momentum_analyzer.py

# 运行特定测试
venv/bin/pytest tests/unit/test_momentum_models.py -v
```

## 性能指标

- **处理时间**: < 1s (2647笔交易)
- **内存使用**: 优化内存管理
- **分析频率**: 可配置，默认60秒一次（每分钟）
- **数据延迟**: 实时处理
- **数据容量**: 支持处理2000+笔交易数据

## Redis数据源

### 前置条件

使用Redis数据源需要：

1. **Redis服务运行**: `redis-server`
2. **trades_window数据**: Redis中需要有`trades_window`键的数据
3. **数据格式**: 每个时间点包含`timestamp`和`price_levels`数据

### 检查Redis状态

```bash
# 检查Redis连接
redis-cli ping

# 检查数据量
redis-cli LLEN trades_window

# 查看最新数据
redis-cli LINDEX trades_window 0
```

### 数据转换逻辑

策略会自动将Redis中的`MinuteTradeData`转换为`Trade`对象：

- 从每个价格水平生成对应的买入/卖出交易
- 保持原始时间戳和价格信息
- 根据买卖成交量分配交易数量

## 风险提示

1. **Redis依赖**: 使用Redis数据源需要Redis服务正常运行
2. **数据质量**: 信号质量取决于trades_window数据的完整性和准确性
3. **参数调优**: 根据实际市场情况调整阈值参数
4. **风险控制**: 建议结合其他技术指标和风险管理措施
5. **回测验证**: 在实盘使用前请进行充分的历史数据回测

## 故障排除

### 常见问题

1. **信号强度为0**: 检查交易数据是否满足最小数量和成交量要求
2. **置信度过低**: 可能是市场波动性过高或数据质量不佳
3. **处理时间过长**: 检查数据量是否过大或系统资源是否充足

### 日志查看

```bash
# 查看策略运行日志
tail -f logs/momentum_strategy.log

# 查看信号历史
tail -f logs/momentum_signals.log
```

## 扩展开发

### 添加新数据源

1. 在 `MomentumStrategyLauncher` 中添加新的数据获取方法
2. 更新配置文件支持新的数据源类型
3. 实现相应的数据格式转换

### 自定义信号逻辑

1. 继承 `ShortTermMomentumAnalyzer` 类
2. 重写 `analyze_momentum` 方法
3. 添加自定义的指标计算逻辑

## 技术支持

如有问题，请查看：
1. 日志文件中的错误信息
2. 单元测试结果
3. 配置文件参数设置

---

**注意**: 本策略仅供学习和研究使用，实盘交易请谨慎评估风险。