# 部署指南

本文档提供了BTC-FDUSD流动性分析智能代理的完整部署指南。

## 🏭 生产环境部署

### 系统要求

**硬件要求：**
- CPU: 4核心以上
- 内存: 8GB以上
- 存储: 50GB SSD
- 网络: 稳定的互联网连接

**软件要求：**
- Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- Python 3.11+
- Redis 6.0+
- Docker (可选)

### 部署架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   App Server 1  │    │   App Server 2  │
│   (Nginx/HAProxy│    │                 │    │                 │
│                 │    │ • Collector     │    │ • Collector     │
│                 │    │ • Analyzer      │    │ • Analyzer      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Redis Cluster │
                    │                 │
                    │ • Master/Slave  │
                    │ • Persistence   │
                    │ • Monitoring    │
                    └─────────────────┘
```

## 📦 安装部署

### 1. 环境准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# 安装Redis
sudo apt install redis-server -y

# 创建应用用户
sudo useradd -m -s /bin/bash strategy-agent
sudo su - strategy-agent
```

### 2. 应用部署

```bash
# 克隆代码
git clone <repository-url> /home/strategy-agent/app
cd /home/strategy-agent/app

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e .
pip install gunicorn supervisor
```

### 3. 配置文件

**生产环境配置 (`config/production.yaml`):**
```yaml
app:
  name: "strategy-agent"
  environment: "production"
  log_level: "INFO"

redis:
  host: "localhost"
  port: 6379
  db: 0
  password: "${REDIS_PASSWORD}"

binance:
  rest_api_base: "https://api.binance.com"
  websocket_base: "wss://stream.binance.com:9443"
  symbol: "BTCFDUSD"
  rate_limit_requests_per_minute: 600
  timeout: 30

data_collector:
  depth_snapshot:
    limit: 5000
    update_interval_seconds: 60
    window_size: 60
  order_flow:
    window_size_minutes: 2880
    price_precision: 1.0
    aggregation_interval_seconds: 60

analyzer:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
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
  file_path: "/var/log/strategy-agent/strategy_agent.log"
  max_file_size_mb: 100
  backup_count: 10
```

### 4. 环境变量

```bash
# 创建环境配置文件
cat > /home/strategy-agent/.env << EOF
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_production_api_key_here

# Redis Configuration
REDIS_PASSWORD=your_redis_password_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Application Configuration
APP_ENV=production
LOG_LEVEL=INFO
EOF

# 设置权限
chmod 600 /home/strategy-agent/.env
```

## 🔧 进程管理

### Supervisor配置

```ini
# /etc/supervisor/conf.d/strategy-agent.conf
[program:strategy-agent-collector]
command=/home/strategy-agent/app/venv/bin/python /home/strategy-agent/app/agent_data_collector.py --config /home/strategy-agent/app/config/production.yaml
directory=/home/strategy-agent/app
user=strategy-agent
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/strategy-agent/collector.log
environment=HOME="/home/strategy-agent",USER="strategy-agent"

[program:strategy-agent-analyzer]
command=/home/strategy-agent/app/venv/bin/python /home/strategy-agent/app/agent_analyzer.py --config /home/strategy-agent/app/config/production.yaml
directory=/home/strategy-agent/app
user=strategy-agent
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/strategy-agent/analyzer.log
environment=HOME="/home/strategy-agent",USER="strategy-agent"
```

**启动服务：**
```bash
# 重新加载配置
sudo supervisorctl reread
sudo supervisorctl update

# 启动服务
sudo supervisorctl start strategy-agent-collector
sudo supervisorctl start strategy-agent-analyzer

# 检查状态
sudo supervisorctl status
```

### Systemd服务配置

```ini
# /etc/systemd/system/strategy-agent-collector.service
[Unit]
Description=Strategy Agent Data Collector
After=network.target redis.service

[Service]
Type=simple
User=strategy-agent
Group=strategy-agent
WorkingDirectory=/home/strategy-agent/app
Environment=PATH=/home/strategy-agent/app/venv/bin
ExecStart=/home/strategy-agent/app/venv/bin/python /home/strategy-agent/app/agent_data_collector.py --config /home/strategy-agent/app/config/production.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/strategy-agent-analyzer.service
[Unit]
Description=Strategy Agent Market Analyzer
After=network.target redis.service

[Service]
Type=simple
User=strategy-agent
Group=strategy-agent
WorkingDirectory=/home/strategy-agent/app
Environment=PATH=/home/strategy-agent/app/venv/bin
ExecStart=/home/strategy-agent/app/venv/bin/python /home/strategy-agent/app/agent_analyzer.py --config /home/strategy-agent/app/config/production.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**启动Systemd服务：**
```bash
# 重新加载systemd配置
sudo systemctl daemon-reload

# 启动并启用服务
sudo systemctl enable strategy-agent-collector
sudo systemctl enable strategy-agent-analyzer
sudo systemctl start strategy-agent-collector
sudo systemctl start strategy-agent-analyzer

# 检查状态
sudo systemctl status strategy-agent-collector
sudo systemctl status strategy-agent-analyzer
```

## 🔍 监控配置

### 日志管理

```bash
# 创建日志目录
sudo mkdir -p /var/log/strategy-agent
sudo chown strategy-agent:strategy-agent /var/log/strategy-agent

# 配置logrotate
sudo cat > /etc/logrotate.d/strategy-agent << EOF
/var/log/strategy-agent/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 strategy-agent strategy-agent
    postrotate
        supervisorctl restart strategy-agent-collector strategy-agent-analyzer
    endscript
}
EOF
```

### 健康检查脚本

```bash
#!/bin/bash
# /home/strategy-agent/health_check.sh

REDIS_CLI="redis-cli"
COLLECTOR_PID=$(pgrep -f "agent_data_collector.py")
ANALYZER_PID=$(pgrep -f "agent_analyzer.py")

# 检查Redis连接
if ! $REDIS_CLI ping > /dev/null 2>&1; then
    echo "CRITICAL: Redis is not responding"
    exit 2
fi

# 检查进程状态
if [ -z "$COLLECTOR_PID" ]; then
    echo "CRITICAL: Data collector is not running"
    exit 2
fi

if [ -z "$ANALYZER_PID" ]; then
    echo "CRITICAL: Analyzer is not running"
    exit 2
fi

# 检查日志中的错误
ERROR_COUNT=$(grep -c "ERROR" /var/log/strategy-agent/strategy_agent.log | tail -100)
if [ "$ERROR_COUNT" -gt 5 ]; then
    echo "WARNING: High error count in logs: $ERROR_COUNT"
    exit 1
fi

echo "OK: All services are running normally"
exit 0
```

### Prometheus监控

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'strategy-agent'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## 🔒 安全配置

### 防火墙设置

```bash
# UFW防火墙配置
sudo ufw allow ssh
sudo ufw allow from 127.0.0.1 to any port 6379  # Redis仅本地访问
sudo ufw allow 8080  # 监控端口
sudo ufw enable
```

### Redis安全

```bash
# /etc/redis/redis.conf
bind 127.0.0.1
port 6379
requirepass your_strong_password_here
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### SSL/TLS配置

```nginx
# /etc/nginx/sites-available/strategy-agent
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🚀 Docker部署

### Dockerfile

```dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 创建应用用户
RUN useradd -m -s /bin/bash strategy-agent

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY pyproject.toml .
COPY README.md .

# 安装Python依赖
RUN pip install --no-cache-dir -e .

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/
COPY agent_*.py .

# 创建日志目录
RUN mkdir -p logs && chown strategy-agent:strategy-agent logs

# 切换到应用用户
USER strategy-agent

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis', port=6379); r.ping()" || exit 1

# 启动命令
CMD ["python", "agent_data_collector.py", "--config", "config/production.yaml"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - strategy-network
    restart: unless-stopped

  data-collector:
    build: .
    command: ["python", "agent_data_collector.py", "--config", "config/production.yaml"]
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    volumes:
      - ./logs:/app/logs
    networks:
      - strategy-network
    depends_on:
      - redis
    restart: unless-stopped

  analyzer:
    build: .
    command: ["python", "agent_analyzer.py", "--config", "config/production.yaml"]
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    volumes:
      - ./logs:/app/logs
    networks:
      - strategy-network
    depends_on:
      - redis
      - data-collector
    restart: unless-stopped

volumes:
  redis_data:

networks:
  strategy-network:
    driver: bridge
```

**部署命令：**
```bash
# 构建和启动
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

## 🔄 备份和恢复

### Redis备份

```bash
#!/bin/bash
# backup_redis.sh

BACKUP_DIR="/backup/redis"
DATE=$(date +%Y%m%d_%H%M%S)
REDIS_CLI="redis-cli -a $REDIS_PASSWORD"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份
$REDIS_cli BGSAVE
sleep 5

# 复制RDB文件
cp /var/lib/redis/dump.rdb $BACKUP_DIR/dump_$DATE.rdb

# 压缩备份
gzip $BACKUP_DIR/dump_$DATE.rdb

# 清理旧备份（保留7天）
find $BACKUP_DIR -name "dump_*.rdb.gz" -mtime +7 -delete

echo "Redis backup completed: dump_$DATE.rdb.gz"
```

### 应用数据备份

```bash
#!/bin/bash
# backup_app.sh

BACKUP_DIR="/backup/app"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/home/strategy-agent/app"

mkdir -p $BACKUP_DIR

# 备份配置文件
tar -czf $BACKUP_DIR/config_$DATE.tar.gz -C $APP_DIR config/

# 备份日志文件
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz -C /var/log/strategy-agent .

# 清理旧备份
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Application backup completed: $DATE"
```

## 📊 性能调优

### Redis优化

```bash
# /etc/redis/redis.conf
# 内存优化
maxmemory 2gb
maxmemory-policy allkeys-lru

# 持久化优化
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes

# 网络优化
tcp-keepalive 300
timeout 0

# 安全优化
requirepass your_strong_password
```

### Python优化

```bash
# 环境变量
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# uvloop (高性能事件循环)
pip install uvloop
```

## 🔧 故障排除

### 常见问题诊断

```bash
# 检查服务状态
systemctl status strategy-agent-collector
systemctl status strategy-agent-analyzer

# 检查Redis连接
redis-cli -a $REDIS_PASSWORD ping

# 查看实时日志
tail -f /var/log/strategy-agent/strategy_agent.log

# 检查网络连接
netstat -tlnp | grep :6379
netstat -tlnp | grep :8080

# 检查系统资源
top -u strategy-agent
free -h
df -h
```

### 性能监控

```bash
# 安装监控工具
pip install psutil

# Python性能监控脚本
python -c "
import psutil
import time
while True:
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    print(f'CPU: {cpu}%, Memory: {memory}%')
    time.sleep(5)
"
```

---

本部署指南涵盖了从开发到生产的完整部署流程。如有问题，请参考日志文件或联系技术支持团队。