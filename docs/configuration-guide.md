# Configuration Management Guide

## 概述

Strategy Agent采用混合配置管理策略，结合YAML配置文件和环境变量，实现了配置的集中管理和敏感信息的安全分离。

## 配置架构

### 配置文件结构

```
config/
└── development.yaml        # 主要配置文件

.env                        # 敏感信息（API密钥等）

src/utils/config.py         # 配置管理模块
```

### 配置层次

1. **config/development.yaml** - 主要配置
   - 应用程序配置
   - Redis连接配置
   - Binance API配置
   - 数据收集参数
   - 分析器设置
   - 日志配置

2. **.env** - 环境变量（敏感信息）
   - API密钥
   - 数据库密码
   - 其他敏感配置

## 配置文件详解

### config/development.yaml

```yaml
app:
  name: "strategy-agent"
  environment: "development"
  log_level: "DEBUG"
  data_retention_days: 30
  max_workers: 4
  batch_size: 1000

redis:
  host: "localhost"
  port: 6379
  db: 0
  decode_responses: true
  socket_timeout: 5
  socket_connect_timeout: 5

binance:
  rest_api_base: "https://api.binance.com"
  websocket_base: "wss://stream.binance.com:9443"
  symbol: "BTCFDUSD"
  rate_limit_requests_per_minute: 1200
  timeout: 30

data_collector:
  depth_snapshot:
    limit: 5000
    update_interval_seconds: 60
    window_size: 60
  order_flow:
    websocket_url: "wss://stream.binance.com:9443/ws/btcfdusd@aggTrade"
    window_size_minutes: 2880
    price_precision: 1.0
    aggregation_interval_seconds: 60

analyzer:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"  # 环境变量引用
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    max_tokens: 4000
    temperature: 0.1
  analysis:
    interval_seconds: 60
    min_order_volume_threshold: 0.01
    support_resistance_threshold: 0.1

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "logs/strategy_agent.log"
  max_file_size_mb: 100
  backup_count: 5
```

### .env 文件

```bash
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_actual_api_key_here
```

## 环境变量扩展

系统支持在YAML配置文件中使用环境变量引用：

### 语法格式

```yaml
# 基本语法
parameter: "${ENV_VAR_NAME}"

# 带默认值（未来版本）
# parameter: "${ENV_VAR_NAME:default_value}"
```

### 使用示例

```yaml
analyzer:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"  # 从环境变量读取
```

### 扩展规则

1. **支持格式**: `${VAR_NAME}`
2. **递归扩展**: 支持在字典、列表、字符串中递归扩展
3. **缺失处理**: 环境变量不存在时使用空字符串并记录警告
4. **日志记录**: 缺失的环境变量会记录警告日志

## 配置加载

### 基本用法

```python
from src.utils.config import Settings

# 从文件加载配置
settings = Settings.load_from_file("config/development.yaml")

# 验证环境变量
settings.validate_required_env_vars()

# 验证配置值
settings.validate_config_values()

# 设置日志
settings.setup_logging()
```

### 配置验证

系统提供多层验证机制：

#### 1. 环境变量验证
```python
settings.validate_required_env_vars()
```
验证必需的环境变量是否存在。

#### 2. 配置值验证
```python
settings.validate_config_values()
```
验证配置参数的合理性：
- 端口范围 (1-65535)
- 超时值 (> 0)
- 符号格式
- API密钥非空
- 日志参数有效性

## 配置最佳实践

### 1. 安全性

- **敏感信息**: 始终使用环境变量存储API密钥、密码等
- **文件权限**: 确保配置文件权限适当 (644)
- **环境隔离**: 不同环境使用不同的配置文件

### 2. 维护性

- **集中管理**: 主要配置集中在YAML文件中
- **文档更新**: 配置变更时同步更新文档
- **版本控制**: 配置文件纳入版本控制（排除.env）

### 3. 部署

- **配置检查**: 部署前验证配置完整性
- **环境测试**: 在测试环境验证配置正确性
- **备份策略**: 重要配置进行备份

## 故障排除

### 常见问题

#### 1. 环境变量未扩展

**症状**: 配置中的`${VAR_NAME}`没有被替换

**解决方案**:
```bash
# 检查环境变量是否设置
echo $DEEPSEEK_API_KEY

# 检查.env文件是否存在
ls -la .env

# 查看日志中的警告信息
journalctl -u strategy-agent-data-collector.service -f
```

#### 2. 配置验证失败

**症状**: 启动时出现配置验证错误

**解决方案**:
```python
# 手动验证配置
from src.utils.config import Settings
settings = Settings.load_from_file("config/development.yaml")
settings.validate_required_env_vars()
settings.validate_config_values()
```

#### 3. 文件路径错误

**症状**: FileNotFoundError: Config file not found

**解决方案**:
```bash
# 检查配置文件路径
ls -la config/development.yaml

# 检查工作目录
pwd
```

### 调试技巧

1. **查看详细错误**:
```bash
# 查看systemd服务日志
sudo journalctl -u strategy-agent-data-collector.service -n 50
```

2. **手动测试配置**:
```python
# 创建测试脚本
python -c "
from src.utils.config import Settings
try:
    settings = Settings.load_from_file('config/development.yaml')
    settings.validate_required_env_vars()
    settings.validate_config_values()
    print('Configuration is valid')
except Exception as e:
    print(f'Configuration error: {e}')
"
```

3. **检查环境变量扩展**:
```python
# 测试环境变量扩展
from src.utils.config import _expand_env_vars
import os

# 设置测试环境变量
os.environ['TEST_VAR'] = 'test_value'

# 测试扩展
result = _expand_env_vars({'param': '${TEST_VAR}'})
print(result)  # 输出: {'param': 'test_value'}
```

## 配置模板

### 开发环境配置

创建 `config/development.yaml`:
```yaml
app:
  environment: "development"
  log_level: "DEBUG"

# 开发环境特定配置
data_collector:
  depth_snapshot:
    update_interval_seconds: 30  # 更频繁的更新
```

### 生产环境配置

创建 `config/production.yaml`:
```yaml
app:
  environment: "production"
  log_level: "INFO"

# 生产环境特定配置
data_collector:
  depth_snapshot:
    update_interval_seconds: 60  # 标准更新频率
```

## 环境变量管理

### 开发环境

```bash
# 创建本地.env文件
cat > .env << EOF
DEEPSEEK_API_KEY=your_development_api_key
EOF
```

### 生产环境

```bash
# 使用系统环境变量或密钥管理系统
export DEEPSEEK_API_KEY=your_production_api_key

# 或通过systemd服务配置
sudo systemctl edit strategy-agent-data-collector.service
# 添加:
# [Service]
# Environment="DEEPSEEK_API_KEY=your_production_api_key"
```

## 总结

Strategy Agent的配置管理系统提供了：

- **✅ 集中化管理**: 主要配置在YAML文件中
- **✅ 安全分离**: 敏感信息通过环境变量管理
- **✅ 自动扩展**: 支持环境变量自动替换
- **✅ 多层验证**: 环境变量和配置值验证
- **✅ 详细文档**: 完整的配置说明和示例

这种配置架构既保证了安全性，又提供了良好的可维护性和灵活性，适合生产环境使用。