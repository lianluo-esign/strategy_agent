# API参考文档

本文档详细描述了BTC-FDUSD流动性分析智能代理的API接口和数据结构。

## 📚 目录

- [核心数据模型](#核心数据模型)
- [Redis数据存储接口](#redis数据存储接口)
- [Binance API集成](#binance-api集成)
- [分析引擎接口](#分析引擎接口)
- [AI客户端接口](#ai客户端接口)
- [配置管理](#配置管理)

## 🏗️ 核心数据模型

### DepthLevel - 深度级别

```python
@dataclass
class DepthLevel:
    price: Decimal      # 价格
    quantity: Decimal   # 数量
```

**说明：** 订单簿中的单个价格级别

**示例：**
```python
level = DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.5'))
```

### DepthSnapshot - 深度快照

```python
@dataclass
class DepthSnapshot:
    symbol: str                    # 交易对
    timestamp: datetime            # 时间戳
    bids: List[DepthLevel]         # 买单列表
    asks: List[DepthLevel]         # 卖单列表
```

**方法：**
- `get_best_bid() -> Optional[Decimal]` - 获取最优买价
- `get_best_ask() -> Optional[Decimal]` - 获取最优卖价
- `get_bid_price_levels() -> List[Decimal]` - 获取所有买价
- `get_ask_price_levels() -> List[Decimal]` - 获取所有卖价

**示例：**
```python
snapshot = DepthSnapshot(
    symbol='BTCFDUSD',
    timestamp=datetime.now(),
    bids=[DepthLevel(Decimal('50000'), Decimal('1.0'))],
    asks=[DepthLevel(Decimal('50001'), Decimal('1.0'))]
)

best_bid = snapshot.get_best_bid()  # Decimal('50000')
best_ask = snapshot.get_best_ask()  # Decimal('50001')
```

### Trade - 交易记录

```python
@dataclass
class Trade:
    symbol: str           # 交易对
    price: Decimal        # 价格
    quantity: Decimal     # 数量
    is_buyer_maker: bool  # 是否买方挂单
    timestamp: datetime   # 时间戳
    trade_id: str         # 交易ID
```

**字段说明：**
- `is_buyer_maker=True`: 主动卖方（吃掉买单）
- `is_buyer_maker=False`: 主动买方（吃掉卖单）

### PriceLevelData - 价格水平数据

```python
@dataclass
class PriceLevelData:
    price_level: Decimal    # 价格水平
    buy_volume: Decimal     # 买入量
    sell_volume: Decimal    # 卖出量
    total_volume: Decimal   # 总量
    delta: Decimal          # 净流入（买入-卖出）
    trade_count: int        # 交易次数
```

**方法：**
- `add_trade(trade: Trade) -> None` - 添加交易记录
- `to_dict() -> Dict` - 转换为字典

### MinuteTradeData - 分钟交易数据

```python
@dataclass
class MinuteTradeData:
    timestamp: datetime                         # 时间戳
    price_levels: Dict[Decimal, PriceLevelData] # 价格水平数据
    max_price_levels: int = 1000               # 最大价格级别数
```

**方法：**
- `add_trade(trade: Trade) -> None` - 添加交易记录
- `cleanup_low_volume_levels(min_volume_threshold) -> None` - 清理低成交量数据
- `to_dict() -> Dict` - 转换为字典

### SupportResistanceLevel - 支撑阻力级别

```python
@dataclass
class SupportResistanceLevel:
    price: Decimal               # 价格
    strength: float              # 强度 (0.0-1.0)
    level_type: str              # 类型 ('support' | 'resistance')
    volume_at_level: Decimal     # 该价位成交量
    confirmation_count: int      # 确认次数
    last_confirmed: Optional[datetime] = None  # 最后确认时间
```

### MarketAnalysisResult - 市场分析结果

```python
@dataclass
class MarketAnalysisResult:
    timestamp: datetime                           # 分析时间
    symbol: str                                   # 交易对
    support_levels: List[SupportResistanceLevel]  # 支撑位
    resistance_levels: List[SupportResistanceLevel] # 阻力位
    poc_levels: List[Decimal]                     # 控制点
    liquidity_vacuum_zones: List[Decimal]         # 流动性真空区
    resonance_zones: List[Decimal]                # 共振区
```

### TradingRecommendation - 交易建议

```python
@dataclass
class TradingRecommendation:
    timestamp: datetime              # 建议时间
    symbol: str                      # 交易对
    action: str                      # 操作 ('BUY' | 'SELL' | 'HOLD')
    price_range: tuple[Decimal, Decimal] # 建议价格区间
    confidence: float                # 置信度 (0.0-1.0)
    reasoning: str                   # 分析逻辑
    risk_level: str                  # 风险等级 ('LOW' | 'MEDIUM' | 'HIGH')
```

## 💾 Redis数据存储接口

### RedisDataStore - Redis数据存储客户端

```python
class RedisDataStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0)
```

#### 深度快照操作

**存储深度快照：**
```python
async def store_depth_snapshot(self, snapshot: DepthSnapshot) -> None
```

**获取最新深度快照：**
```python
def get_latest_depth_snapshot(self) -> Optional[DepthSnapshot]
```

**存储键格式：** `depth_snapshot_5000`
**数据结构：** List (滑动窗口，保留最近60个)

#### 交易数据操作

**存储分钟交易数据：**
```python
async def store_minute_trade_data(self, trade_data: MinuteTradeData) -> None
```

**获取最近交易数据：**
```python
def get_recent_trade_data(self, minutes: int = 60) -> List[MinuteTradeData]
```

**存储键格式：** `trades_window`
**数据结构：** List (滑动窗口，保留最近2880分钟)

#### 分析结果操作

**存储分析结果：**
```python
async def store_analysis_result(self, result: MarketAnalysisResult) -> None
```

**获取最新分析结果：**
```python
def get_latest_analysis_result(self) -> Optional[MarketAnalysisResult]
```

**存储键格式：** `analysis_results:{timestamp}`
**数据结构：** String (JSON格式，过期时间1小时)

#### 工具方法

**测试连接：**
```python
def test_connection(self) -> bool
```

**获取数据计数：**
```python
def get_depth_snapshot_count(self) -> int
def get_trade_window_count(self) -> int
```

**清理数据：**
```python
def clear_all_data(self) -> None
```

**关闭连接：**
```python
async def close(self) -> None
```

## 🌐 Binance API集成

### BinanceAPIClient - REST API客户端

```python
class BinanceAPIClient:
    def __init__(self, base_url: str = BINANCE_REST_API_BASE, timeout: int = 30)
```

#### 获取深度快照

```python
async def get_depth_snapshot(
    self,
    symbol: str = BTC_FDUSD_SYMBOL,
    limit: int = DEPTH_SNAPSHOT_LIMIT
) -> Optional[DepthSnapshot]
```

**参数：**
- `symbol`: 交易对 (默认: 'BTCFDUSD')
- `limit`: 深度级别 (默认: 5000)

**返回：** DepthSnapshot对象或None

**API端点：** `GET /api/v3/depth`

#### 会话管理

```python
async def get_async_session(self) -> aiohttp.ClientSession
async def close_async_session(self) -> None
```

### BinanceWebSocketClient - WebSocket客户端

```python
class BinanceWebSocketClient:
    def __init__(self, symbol: str = BTC_FDUSD_SYMBOL)
```

#### 连接管理

```python
async def connect(self) -> bool
async def disconnect(self) -> None
async def listen_trades(self, callback) -> None
```

#### 连接状态

```python
async def ping(self) -> bool
```

**WebSocket URL：** `wss://stream.binance.com:9443/ws/btcfdusd@aggTrade`

## 🔍 分析引擎接口

### DepthSnapshotAnalyzer - 深度快照分析器

```python
class DepthSnapshotAnalyzer:
    def __init__(self, min_volume_threshold: float = 0.1, price_zone_size: float = 0.50)
```

#### 支撑阻力分析

```python
def analyze_support_resistance(
    self,
    snapshot: DepthSnapshot,
    lookback_levels: int = 100
) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]
```

**返回：** (支撑位列表, 阻力位列表)

#### 流动性真空区识别

```python
def identify_liquidity_vacuum_zones(
    self,
    snapshot: DepthSnapshot,
    price_range: Optional[Tuple[Decimal, Decimal]] = None
) -> List[Decimal]
```

**返回：** 真空区价格列表

### OrderFlowAnalyzer - 订单流分析器

```python
class OrderFlowAnalyzer:
    def __init__(self, analysis_window_minutes: int = 180)
```

#### 订单流分析

```python
def analyze_order_flow(
    self,
    trade_data_list: List[MinuteTradeData],
    support_levels: List[SupportResistanceLevel],
    resistance_levels: List[SupportResistanceLevel]
) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel], List[Decimal]]
```

**返回：** (确认的支撑位, 确认的阻力位, POC级别)

#### 控制点识别

```python
def _find_poc_levels(self, trade_data_list: List[MinuteTradeData]) -> List[Decimal]
```

### MarketAnalyzer - 综合市场分析器

```python
class MarketAnalyzer:
    def __init__(
        self,
        min_volume_threshold: float = 0.1,
        analysis_window_minutes: int = 180
    )
```

#### 综合分析

```python
def analyze_market(
    self,
    snapshot: Optional[DepthSnapshot],
    trade_data_list: List[MinuteTradeData],
    symbol: str = "BTCFDUSD"
) -> MarketAnalysisResult
```

#### 共振区识别

```python
def _find_resonance_zones(self, result: MarketAnalysisResult) -> List[Decimal]
```

## 🤖 AI客户端接口

### DeepSeekClient - DeepSeek AI客户端

```python
class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        temperature: float = 0.1
    )
```

#### 市场数据分析

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def analyze_market_data(
    self,
    analysis_result: MarketAnalysisResult,
    symbol: str = "BTCFDUSD"
) -> Optional[TradingRecommendation]
```

**工具调用：**
- `depth_snapshot_analysis` - 深度快照分析
- `orderflow_analysis` - 订单流分析

#### 连接管理

```python
async def close(self) -> None
```

## ⚙️ 配置管理

### Settings - 配置类

```python
class Settings(BaseSettings):
    app: AppConfig
    redis: RedisConfig
    binance: BinanceConfig
    data_collector: DataCollectorConfig
    analyzer: AnalyzerConfig
    logging: LoggingConfig
```

#### 加载配置

```python
@classmethod
def load_from_file(cls, config_path: str) -> "Settings"
```

#### 日志设置

```python
def setup_logging(self) -> None
```

### 配置组件

#### AppConfig - 应用配置

```python
class AppConfig(BaseModel):
    name: str = "strategy-agent"
    environment: str = "development"
    log_level: str = "DEBUG"
```

#### RedisConfig - Redis配置

```python
class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
```

#### BinanceConfig - Binance配置

```python
class BinanceConfig(BaseModel):
    rest_api_base: str = "https://api.binance.com"
    websocket_base: str = "wss://stream.binance.com:9443"
    symbol: str = "BTCFDUSD"
    rate_limit_requests_per_minute: int = 1200
    timeout: int = 30
```

#### DeepSeekConfig - DeepSeek配置

```python
class DeepSeekConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_tokens: int = 4000
    temperature: float = 0.1
```

## 🔧 错误处理

### 异常类型

```python
# 连接错误
class RedisConnectionError(Exception): pass
class BinanceAPIError(Exception): pass
class WebSocketConnectionError(Exception): pass

# 数据错误
class DataValidationError(Exception): pass
class InsufficientDataError(Exception): pass

# 分析错误
class AnalysisError(Exception): pass
class AIServiceError(Exception): pass
```

### 错误响应格式

```python
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid price value",
        "details": {
            "field": "price",
            "value": -100.0,
            "constraint": "price > 0"
        }
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456"
}
```

## 📊 状态码

### HTTP状态码

- `200 OK` - 请求成功
- `400 Bad Request` - 请求参数错误
- `401 Unauthorized` - 认证失败
- `429 Too Many Requests` - 请求频率超限
- `500 Internal Server Error` - 服务器内部错误
- `503 Service Unavailable` - 服务不可用

### 业务状态码

- `SUCCESS` - 操作成功
- `INSUFFICIENT_DATA` - 数据不足
- `ANALYSIS_FAILED` - 分析失败
- `SERVICE_UNAVAILABLE` - 服务不可用

## 📝 使用示例

### 完整分析流程示例

```python
import asyncio
from src.core.redis_client import RedisDataStore
from src.core.analyzers import MarketAnalyzer
from src.utils.config import Settings

async def main():
    # 加载配置
    settings = Settings.load_from_file("config/production.yaml")

    # 初始化组件
    redis_store = RedisDataStore(
        host=settings.redis.host,
        port=settings.redis.port,
        db=settings.redis.db
    )

    analyzer = MarketAnalyzer(
        min_volume_threshold=settings.analyzer.analysis.min_order_volume_threshold
    )

    # 获取数据
    snapshot = redis_store.get_latest_depth_snapshot()
    trade_data = redis_store.get_recent_trade_data(minutes=180)

    # 执行分析
    if snapshot and trade_data:
        result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data,
            symbol="BTCFDUSD"
        )

        print(f"分析完成：")
        print(f"支撑位数量: {len(result.support_levels)}")
        print(f"阻力位数量: {len(result.resistance_levels)}")
        print(f"共振区数量: {len(result.resonance_zones)}")

    # 清理
    await redis_store.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 自定义分析器示例

```python
from src.core.analyzers import DepthSnapshotAnalyzer
from src.core.models import DepthSnapshot, DepthLevel
from decimal import Decimal
from datetime import datetime

# 创建分析器
analyzer = DepthSnapshotAnalyzer(
    min_volume_threshold=0.5,
    price_zone_size=1.0
)

# 创建测试数据
snapshot = DepthSnapshot(
    symbol='BTCFDUSD',
    timestamp=datetime.now(),
    bids=[
        DepthLevel(Decimal('50000'), Decimal('10.0')),  # 大单
        DepthLevel(Decimal('49999'), Decimal('1.0')),
        DepthLevel(Decimal('49998'), Decimal('0.5')),
    ],
    asks=[
        DepthLevel(Decimal('50001'), Decimal('0.8')),
        DepthLevel(Decimal('50002'), Decimal('2.0')),
        DepthLevel(Decimal('50003'), Decimal('8.0')),   # 大单
    ]
)

# 执行分析
support, resistance = analyzer.analyze_support_resistance(snapshot)

print("支撑位:")
for level in support:
    print(f"  价格: ${level.price}, 强度: {level.strength:.2f}")

print("阻力位:")
for level in resistance:
    print(f"  价格: ${level.price}, 强度: {level.strength:.2f}")
```

---

本API参考文档提供了系统所有主要接口的详细说明。如需更多示例或有其他问题，请参考源代码或联系开发团队。