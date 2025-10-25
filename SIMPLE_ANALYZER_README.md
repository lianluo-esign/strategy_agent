# 简化市场分析器

## 概述

这是一个简化的市场分析器，专注于核心功能，避免了原来系统中复杂的重复分析工具。

## 核心功能

### 1. Redis深度快照数据读取和聚合处理
- 从Redis缓存读取最新的深度快照数据
- 按照配置的聚合精度（例如$1.0精度）进行价格聚合
- 将分散的订单簿数据聚合成有意义的流动性层级

### 2. DeepSeek LLM支撑阻力分析
- 将聚合后的订单簿数据打包成结构化prompt
- 提交给DeepSeek LLM进行专业的市场结构分析
- 在info日志中详细打印分析结果，包括：
  - 🟢 买盘支撑区域（价格区间、强度、特征）
  - 🔻 卖盘阻力区域（价格区间、强度、特征）
  - ⚖️ 市场平衡状态（买卖力量对比分析）
  - 📍 关键价位（重要价格水平及作用）

### 3. 挂单分布可视化
- 使用visualization功能生成订单簿分布图
- 直观显示买卖盘流动性分布
- 保存为PNG图片文件

## 使用方法

### 1. 配置文件
确保 `config/development.yaml` 中的配置正确：

```yaml
analyzer:
  deepseek:
    enable: true
    api_key: "your_api_key_here"
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    max_tokens: 4000
    temperature: 0.1
  price_aggregation:
    precision: 1.0  # $1精度聚合
    enabled: true
    max_price_levels: 5000
  visualization:
    enabled: true
    output_base_path: "./visualizations/order_book/"
  analysis:
    interval_seconds: 300  # 5分钟分析间隔
```

### 2. 启动简化分析器

```bash
# 使用简化分析器脚本
python simple_analyzer.py

# 或者直接运行模块
python -m src.agents.simple_analyzer_agent

# 或者指定配置文件
python simple_analyzer.py --config config/production.yaml
```

### 3. 查看分析结果

分析器运行后，你将在日志中看到类似以下的输出：

```
2025-10-26 01:49:29,223 - INFO - === BTCFDUSD DeepSeek LLM 支撑阻力分析 ===
2025-10-26 01:49:29,223 - INFO - 🟢 买盘支撑区域:
2025-10-26 01:49:29,223 - INFO -   支撑 1: $111,960.00-$111,940.00 | 强度: 85 | 特征: 该区域集中了显著的买盘流动性...
2025-10-26 01:49:29,224 - INFO - 🔻 卖盘阻力区域:
2025-10-26 01:49:29,224 - INFO -   阻力 1: $112,010.00-$112,030.00 | 强度: 80 | 特征: 在$112,010.00和$112,030.00分别有1.51的卖盘挂单...
2025-10-26 01:49:29,224 - INFO - ⚖️  市场平衡状态: 买盘强势
2025-10-26 01:49:29,224 - INFO - 📍 关键价位:
2025-10-26 01:49:29,224 - INFO -   关键价位 1: $111,960.00 | 作用: 主要支撑 | 重要性: 最强支撑价位...
```

### 4. 查看可视化结果

订单簿分布图将保存在 `visualizations/order_book/` 目录下，文件名格式为：
`order_book_distribution_BTCFDUSD_YYYYMMDD_HHMMSS.png`

## 文件结构

```
src/core/
├── simple_market_analyzer.py     # 核心分析逻辑
├── price_aggregator.py          # 价格聚合工具
└── deepseek_analyzer.py         # DeepSeek LLM分析器

src/agents/
└── simple_analyzer_agent.py     # 简化分析器代理

simple_analyzer.py               # 启动脚本
```

## 与原系统的区别

### 原系统问题：
- 有多个重复的分析工具（LiquidityPeaksAnalyzer, NormalDistributionMarketAnalyzer等）
- 复杂的异步集成导致超时问题
- 分析结果混乱，同时显示交易指令和市场分析
- 流程不够清晰

### 简化系统优势：
- **单一分析流程**：只有一条清晰的分析链路
- **专注核心功能**：只做读取→聚合→分析→可视化
- **清晰的日志输出**：DeepSeek分析结果在info日志中清晰显示
- **无重复工具**：移除了所有冗余的分析器
- **稳定的性能**：避免了复杂的异步问题

## 环境要求

- Python 3.8+
- Redis服务器运行
- DeepSeek API密钥
- 依赖包：`pip install -r requirements.txt`

## 故障排除

1. **Redis连接失败**：确保Redis服务器运行且配置正确
2. **DeepSeek分析超时**：API调用可能需要30-60秒，这是正常的
3. **没有深度快照数据**：确保数据收集器正在运行并向Redis写入数据
4. **可视化失败**：检查matplotlib依赖和输出目录权限

## 停止分析器

使用 `Ctrl+C` 优雅停止分析器。