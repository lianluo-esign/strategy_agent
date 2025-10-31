# Discord配置指南

## 配置Discord Webhook通知

优化版Agent Analyzer现在支持从配置文件中直接读取所有参数，包括DeepSeek API密钥和Discord webhook URL。

## 直接在配置文件中配置所有参数

### 完整配置示例

```yaml
# config/development.yaml
app:
  name: "strategy-agent"
  environment: "development"

redis:
  host: "localhost"
  port: 6379
  db: 0

binance:
  rest_api_base: "https://api.binance.com"
  websocket_base: "wss://stream.binance.com:9443"
  symbol: "BTCFDUSD"

analyzer:
  deepseek:
    enable: true
    api_key: "sk-your-deepseek-api-key"  # 直接填写API密钥
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    max_tokens: 6000
    temperature: 0.1
    timeout: 90
    max_retries: 3

  discord:
    enable: true
    webhook_url: "https://discord.com/api/webhooks/1433882831775338558/ZlPlAiCFA49TidMxyiapXzJ89MDkA8gYy5uOPxZRr7-orOwRa-71Hc_79Fa6D7SU2K6C"  # 直接填写Discord webhook URL
    timeout: 30
    max_retries: 3

  analysis:
    interval_seconds: 300  # 分析间隔：5分钟

  price_aggregation:
    precision: 1.0
    enabled: true
    max_price_levels: 5000

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 配置参数说明

### DeepSeek配置
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | boolean | true | 是否启用Deepseek分析 |
| `api_key` | string | "" | Deepseek API密钥 |
| `base_url` | string | "https://api.deepseek.com/v1" | API基础URL |
| `model` | string | "deepseek-chat" | 使用的模型 |
| `max_tokens` | integer | 4000 | 最大令牌数 |
| `temperature` | float | 0.1 | 温度参数 |
| `timeout` | integer | 60 | 请求超时时间（秒） |
| `max_retries` | integer | 3 | 最大重试次数 |

### Discord配置
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | boolean | false | 是否启用Discord通知 |
| `webhook_url` | string | "" | Discord Webhook URL |
| `timeout` | integer | 30 | HTTP请求超时时间（秒） |
| `max_retries` | integer | 3 | 最大重试次数 |

## 使用示例

### 运行程序

```bash
# 使用包含所有配置的文件
python agent_analyzer_optimized.py --config config/development_with_discord.yaml

# 或者使用原始development.yaml文件
python agent_analyzer_optimized.py --config config/development.yaml
```

## 创建Discord Webhook

1. **在Discord服务器中**：
   - 进入服务器设置
   - 选择"集成"
   - 选择"Webhooks"
   - 点击"新建Webhook"

2. **配置Webhook**：
   - 输入Webhook名称（如：市场分析通知）
   - 选择要发送通知的频道
   - 复制生成的Webhook URL

3. **权限设置**：
   - 确保Webhook有发送消息的权限
   - 可以选择特定的消息类型（嵌入消息、附件等）

## 通知格式

Discord通知将包含：
- 📈 趋势判断（震荡/看涨/看跌等）
- 🎯 置信度
- 💪 支撑/阻力强度可视化
- 📝 详细分析原因
- ⏱️ 处理时间
- 📊 数据统计信息

## 配置文件示例

### 最小配置示例
```yaml
analyzer:
  deepseek:
    enable: true
    api_key: "your-api-key"
  discord:
    enable: true
    webhook_url: "https://discord.com/api/webhooks/your-webhook"
  analysis:
    interval_seconds: 300
```

### 完整配置示例
```yaml
analyzer:
  deepseek:
    enable: true
    api_key: "sk-2ce70cb7d88e44c19d65524b990b192b"
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    max_tokens: 6000
    temperature: 0.1
    timeout: 90
    max_retries: 3

  discord:
    enable: true
    webhook_url: "https://discord.com/api/webhooks/1433882831775338558/ZlPlAiCFA49TidMxyiapXzJ89MDkA8gYy5uOPxZRr7-orOwRa-71Hc_79Fa6D7SU2K6C"
    timeout: 30
    max_retries: 3

  analysis:
    interval_seconds: 300
    min_order_volume_threshold: 0.01
    support_resistance_threshold: 0.1
```

## 运行模式

### 单次运行模式
```bash
python agent_analyzer_optimized.py --single-run --config config/development_with_discord.yaml
```

### 持续运行模式
```bash
python agent_analyzer_optimized.py --config config/development_with_discord.yaml
```

## 故障排除

### 1. Discord通知未发送
- 检查`discord.enable`是否为`true`
- 验证`webhook_url`是否正确
- 查看日志中的Discord相关错误信息

### 2. Deepseek API调用失败
- 检查`deepseek.enable`是否为`true`
- 验证`api_key`是否正确
- 检查网络连接和API访问权限

### 3. 配置文件解析失败
- 检查YAML文件语法是否正确
- 验证文件路径是否存在
- 查看配置解析错误日志

### 4. Webhook URL无效
- Discord返回404：检查URL是否正确复制
- Discord返回403：检查Webhook权限
- Discord返回405：检查HTTP方法（系统使用POST）

## 安全注意事项

1. **保护敏感信息**：
   - 不要将包含API密钥的配置文件提交到公共代码仓库
   - 使用版本控制系统时，考虑使用`.gitignore`排除敏感配置
   - 定期轮换API密钥

2. **权限管理**：
   - 为Discord Webhook创建专门的频道
   - 限制Webhook的权限范围
   - 监控Webhook的使用情况

3. **配置文件管理**：
   - 保留配置文件的备份
   - 使用版本控制管理配置变更
   - 在不同环境中使用不同的配置文件

## 配置文件位置

- **开发环境**: `config/development.yaml`
- **示例配置**: `config/development_with_discord.yaml`
- **生产环境**: `config/production.yaml` (需要创建)

所有参数现在都直接从配置文件中读取，无需设置任何环境变量。