# BTC-FDUSD 流动性分析智能代理

基于Binance BTC-FDUSD现货市场的深度订单簿和订单流数据分析的智能做市系统。

## 🎯 系统概述

本系统通过三层分析架构实现专业的市场分析和做市策略：

1. **静态支撑/阻力分析** - 基于5000层深度快照识别关键价格水平
2. **动态订单流确认** - 通过48小时交易数据验证市场结构
3. **AI驱动决策** - 使用DeepSeek AI生成智能交易建议

## 🏗️ 系统架构

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Collector    │    │      Redis Store    │    │   Market Analyzer   │
│                     │    │                     │    │                     │
│ • Binance REST API  │───▶│ • Depth Snapshots   │◀───│ • Technical Analysis│
│ • WebSocket Stream  │    │ • Trade Window      │    │ • AI Integration    │
│ • Data Aggregation  │    │ • Analysis Results  │    │ • Recommendations  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Redis Server
- DeepSeek API Key

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd strategy_agent
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -e .
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，添加您的 DeepSeek API Key
```

5. **启动Redis服务器**
```bash
redis-server
```

### 运行系统

1. **启动数据收集进程**
```bash
python agent_data_collector.py --config config/development.yaml
```

2. **启动市场分析进程**
```bash
python agent_analyzer.py --config config/development.yaml
```

## 📊 核心功能

### 数据收集模块 (Data Collector)

**功能特性：**
- 每60秒获取5000层深度快照
- 实时WebSocket订单流数据收集
- 1分钟价格水平聚合
- 48小时滑动窗口数据管理
- **历史数据持久化**: 自动将过期数据序列化到本地磁盘

**关键指标：**
- 深度快照更新频率：60秒
- 订单流聚合精度：$1.0
- 数据保留周期：48小时 (Redis) + 无限 (文件存储)

### 市场分析模块 (Market Analyzer)

**静态分析功能：**
- 大单墙识别（支撑/阻力）
- 高密度价格区间检测
- 流动性真空区识别

**动态分析功能：**
- 控制点(POC)计算
- 订单流确认验证
- 共振区域识别

**AI智能分析：**
- DeepSeek AI集成
- 多维度信号融合
- 风险评估和置信度

## 🔧 配置说明

### 配置文件结构

```yaml
app:
  name: "strategy-agent"
  environment: "development"
  log_level: "DEBUG"

redis:
  host: "localhost"
  port: 6379
  db: 0
  storage_dir: "storage"  # 历史数据存储目录

binance:
  symbol: "BTCFDUSD"
  timeout: 30

data_collector:
  depth_snapshot:
    limit: 5000
    update_interval_seconds: 60
    window_size: 60
  order_flow:
    window_size_minutes: 2880  # 48 hours
    price_precision: 1.0

analyzer:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
    temperature: 0.1
  analysis:
    interval_seconds: 60
    min_order_volume_threshold: 0.01
```

### 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `depth_snapshot.limit` | 5000 | 订单簿深度级别 |
| `order_flow.price_precision` | 1.0 | 价格聚合精度($) |
| `analysis.interval_seconds` | 60 | 分析间隔(秒) |
| `deepseek.temperature` | 0.1 | AI创造性参数 |

## 📈 数据模型

### 深度快照 (DepthSnapshot)
```python
@dataclass
class DepthSnapshot:
    symbol: str
    timestamp: datetime
    bids: List[DepthLevel]
    asks: List[DepthLevel]
```

### 分钟交易数据 (MinuteTradeData)
```python
@dataclass
class MinuteTradeData:
    timestamp: datetime
    price_levels: Dict[Decimal, PriceLevelData]
    max_price_levels: int = 1000
```

### 分析结果 (MarketAnalysisResult)
```python
@dataclass
class MarketAnalysisResult:
    timestamp: datetime
    symbol: str
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    poc_levels: List[Decimal]
    resonance_zones: List[Decimal]
```

## 💾 历史数据持久化

### 功能概述
系统自动将超过48小时历史的交易数据序列化到本地磁盘，确保数据的长期保存和分析价值。

### 核心特性
- **异步文件操作**: 使用`aiofiles`实现非阻塞文件I/O
- **并发写入**: 多个文件同时写入，提高性能
- **时间戳命名**: 文件格式`trades_YYYYMMDD_HHMM.json`
- **错误容错**: 文件写入失败不影响主流程

### 文件结构示例
```
storage/
├── trades_20241023_1430.json
├── trades_20241023_1431.json
├── trades_20241023_1432.json
└── ...
```

### 配置参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `redis.storage_dir` | "storage" | 历史数据存储目录 |

详细文档请参考: [Trade Data Persistence](docs/trade_data_persistence.md)

## 🧪 测试

### 运行测试套件
```bash
# 运行所有测试
pytest

# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 测试覆盖率报告
pytest --cov=. --cov-report=html
```

### 测试覆盖率
- 单元测试覆盖率：90%+
- 集成测试：关键数据流程验证
- Mock测试：外部依赖隔离

## 📋 API文档

### Redis数据存储

**深度快照数据：**
- Key: `depth_snapshot_5000`
- Type: List (滑动窗口)
- 保留：最近60个快照

**交易数据窗口：**
- Key: `trades_window`
- Type: List (滑动窗口)
- 保留：最近2880分钟(48小时)
- **过期处理**: 自动序列化到`storage/`目录为JSON文件

**分析结果：**
- Key: `analysis_results:{timestamp}`
- Type: String (JSON)
- 过期：1小时

### Binance API集成

**REST API端点：**
```
GET /api/v3/depth?symbol=BTCFDUSD&limit=5000
```

**WebSocket流：**
```
wss://stream.binance.com:9443/ws/btcfdusd@aggTrade
```

## 🔍 监控和日志

### 日志配置
```python
# 日志级别
DEBUG: 详细调试信息
INFO: 一般运行信息
WARNING: 警告信息
ERROR: 错误信息

# 日志文件
logs/strategy_agent.log (主日志)
logs/trading_recommendations_btcfdusd.log (交易建议)
```

### 状态监控
```python
# 数据收集器状态
{
    'is_running': True,
    'websocket_connected': True,
    'depth_snapshots_count': 45,
    'trade_window_count': 120
}

# 分析器状态
{
    'is_running': True,
    'redis_connected': True,
    'last_analysis': '2024-01-01T12:00:00'
}
```

## 🛡️ 安全考虑

### API密钥管理
- 使用环境变量存储敏感信息
- 配置文件支持变量替换
- 生产环境使用密钥管理服务

### 网络安全
- HTTPS/WSS加密通信
- 连接超时配置
- 速率限制保护

### 数据验证
- 输入数据严格验证
- 价格和数量范围检查
- 恶意数据过滤

## 🔄 优雅关闭机制

### 问题解决
修复了运行数据收集代理时使用 Ctrl+C 导致进程僵死的问题。

### 核心特性
- **立即响应**: Ctrl+C 后 100ms 内开始关闭流程
- **任务取消**: 主动取消所有运行中的异步任务
- **数据保护**: 确保关闭前保存所有重要数据
- **资源清理**: 正确关闭 WebSocket、Redis、HTTP 连接
- **超时保护**: 多层超时机制防止无限期阻塞

### 使用方法
```bash
# 正常启动
python agent_data_collector.py --config config/development.yaml

# 优雅关闭
Ctrl+C  # 立即响应，3-5秒内完成关闭
```

### 技术实现
- 跨线程信号处理使用 `call_soon_threadsafe()`
- 分层关闭策略确保正确的资源清理顺序
- 所有异步操作支持取消和超时
- 完整的错误处理和日志记录

详细文档参考: [优雅关闭功能](docs/graceful_shutdown.md)

## 🚨 故障排除

### 常见问题

**1. Redis连接失败**
```bash
# 检查Redis服务状态
redis-cli ping

# 检查配置文件中的Redis设置
host: localhost
port: 6379
db: 0
```

**2. WebSocket连接断开**
```bash
# 检查网络连接
curl -I https://stream.binance.com:9443

# 查看日志中的连接错误
tail -f logs/strategy_agent.log | grep "WebSocket"
```

**3. DeepSeek API错误**
```bash
# 验证API密钥
export DEEPSEEK_API_KEY="your_key_here"

# 检查API配额
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
     https://api.deepseek.com/v1/models
```

### 性能优化

**内存使用优化：**
- 价格水平数据限制：1000个/分钟
- 低成交量数据自动清理
- Redis内存使用监控

**网络优化：**
- HTTP连接池复用
- WebSocket心跳检测
- 请求重试机制

## 📚 开发指南

### 代码结构
```
src/
├── core/           # 核心业务逻辑
│   ├── models.py   # 数据模型
│   ├── analyzers.py # 分析引擎
│   └── redis_client.py # Redis客户端
├── utils/          # 工具模块
│   ├── config.py   # 配置管理
│   ├── binance_client.py # Binance API
│   └── ai_client.py # AI客户端
└── agents/         # 智能代理
    ├── data_collector.py # 数据收集
    └── analyzer.py # 市场分析
```

### 代码质量标准
- **代码风格**: Ruff格式化检查
- **类型注解**: 100%类型覆盖
- **测试覆盖率**: 90%+
- **文档字符串**: 公共API完整文档

### 开发工作流
1. 创建功能分支
2. 编写代码和测试
3. 运行质量检查
4. 提交代码审查
5. 合并主分支

## 🤝 贡献指南

### 提交代码
1. Fork项目仓库
2. 创建功能分支
3. 编写测试用例
4. 确保所有测试通过
5. 提交Pull Request

### 报告问题
- 使用GitHub Issues报告bug
- 提供详细的错误日志
- 包含复现步骤
- 标明环境信息

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 📞 支持

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- 技术文档：`docs/`目录
- API参考：`docs/api.md`

---

**注意**: 本系统仅用于数据分析和研究目的，不构成投资建议。使用本系统进行实际交易存在风险，请谨慎使用。