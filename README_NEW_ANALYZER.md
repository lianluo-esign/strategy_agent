# 🚀 新版简化市场分析器

基于用户要求的三步分析流程实现的BTC-FDUSD流动性分析agent。

## 📋 功能特点

### 三步分析流程
1. **初始化** - 加载配置，连接Redis和DeepSeek
2. **数据读取与聚合** - 从Redis读取历史数据，聚合orderbook和trades_window
3. **AI分析** - 将聚合数据发送给DeepSeek进行动态分析

### 标准输出格式
```json
{
  "grid_delta": 2.0,      // 范围: 0.1-100.0
  "grid_quantity": 0.001, // 范围: 0.0001-10.0
  "active_side": "Buy"    // 选项: "Buy" 或 "Sell"
}
```

## 🚦 快速开始

### 1. 设置环境变量
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

### 2. 运行分析器

#### 持续运行模式（推荐）
```bash
# 使用开发配置
python agent_analyzer.py

# 使用生产配置
python agent_analyzer.py --config config/production.yaml
```

#### 单次测试模式
```bash
# 运行一次分析并退出
python agent_analyzer.py --single-run
```

### 3. 查看帮助
```bash
python agent_analyzer.py --help
```

## 📊 日志输出示例

```
🚀 Starting Simplified Market Analyzer Agent
📋 3-Step Analysis Flow:
   1. Initialize components
   2. Read and aggregate Redis data
   3. Send to DeepSeek for analysis

✅ Redis connection established
🔄 Starting analysis loop for BTCFDUSD (interval: 60s)

=== 📊 BTCFDUSD Simplified Analysis Summary ===
🔍 Analysis Mode: Simplified 3-Step Analysis
📊 Market Data: 2128 bid levels, 1873 ask levels
📈 Volume Profile: 312 price levels, total_volume=12806.08
✅ Trading Parameters Generated:
   💰 Grid Delta: 2.0
   📊 Grid Quantity: 0.001
   🎯 Active Side: Buy
==================================================
🎯 Trading Parameters Ready: {'grid_delta': 2.0, 'grid_quantity': 0.001, 'active_side': 'Buy'}
```

## ⚙️ 配置要求

### 必需配置
- **DeepSeek API Key**: 必须设置 `DEEPSEEK_API_KEY` 环境变量
- **Redis连接**: 配置文件中的Redis主机、端口和数据库
- **交易对**: 默认为 `BTCFDUSD`

### 配置文件示例
```yaml
# config/development.yaml
redis:
  host: "localhost"
  port: 6379
  db: 0

binance:
  symbol: "BTCFDUSD"

analyzer:
  analysis:
    interval_seconds: 60

  deepseek:
    enable: true
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    temperature: 0.1
    timeout: 90
```

## 🛠️ 核心技术特性

### 鲁棒的JSON解析
- 4层回退策略确保即使DeepSeek返回格式错误的响应也能正常工作
- 参数范围验证和业务逻辑检查
- 预编译正则表达式提高性能

### 动态分析
- 移除了硬编码的JSON示例
- AI基于真实市场数据生成分析结果
- 每次调用都可能产生不同的交易参数

### 生产就绪
- 完整的日志记录和错误处理
- 优雅关闭和信号处理
- 自动重试和故障恢复

## 📁 项目结构

```
agent_analyzer.py                    # 🚀 新入口文件
├── src/core/
│   ├── result_validator.py          # 结果验证器
│   ├── simplified_market_analyzer.py # 简化市场分析器
│   ├── unified_deepseek_analyzer.py # DeepSeek AI分析器
│   └── redis_client.py             # Redis客户端
├── tests/unit/
│   ├── test_result_validator.py     # 验证器测试
│   └── test_simplified_market_analyzer.py # 分析器测试
└── config/
    ├── development.yaml             # 开发配置
    └── production.yaml             # 生产配置
```

## 🧪 测试

### 运行单元测试
```bash
# 测试结果验证器
PYTHONPATH=/path/to/project venv/bin/pytest tests/unit/test_result_validator.py -v

# 测试简化分析器
PYTHONPATH=/path/to/project venv/bin/pytest tests/unit/test_simplified_market_analyzer.py -v
```

### 验证功能
```bash
# 验证验证器功能
python test_simplified_analyzer.py
```

## 🔧 故障排除

### 常见问题

1. **DeepSeek API Key错误**
   ```
   ❌ DeepSeek API key is required for simplified analyzer
   ```
   解决方案: 设置环境变量 `export DEEPSEEK_API_KEY="your-key"`

2. **Redis连接失败**
   ```
   Failed to connect to Redis. Exiting...
   ```
   解决方案: 检查Redis服务是否运行，配置文件中的连接参数是否正确

3. **无数据可用**
   ```
   ℹ️  No data available for simplified analysis
   ```
   解决方案: 确保Redis中有深度快照和交易数据

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=/path/to/project
python agent_analyzer.py --single-run
```

## 🎯 预期结果

每次分析都会输出：
- ✅ 成功时：具体的交易参数（grid_delta, grid_quantity, active_side）
- ❌ 失败时：详细的错误信息和原因
- ℹ️  无数据时：提示数据不可用

系统会持续运行，按照配置的时间间隔（默认60秒）重复执行三步分析流程。