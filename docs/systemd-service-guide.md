# Strategy Agent Data Collector Systemd 服务指南

## 概述

本文档描述了如何使用systemd管理Strategy Agent Data Collector服务的启停和监控。

## 文件结构

```
systemd/
├── strategy-agent-data-collector.service    # 主服务配置
├── strategy-agent-data-collector-health.service  # 健康检查服务
├── strategy-agent-data-collector.timer      # 定时健康检查
├── logrotate.d/
│   └── strategy-agent-data-collector        # 日志轮转配置
└── install.sh                               # 安装脚本
```

## 快速开始

### 1. 安装服务

```bash
# 进入项目目录
cd /home/jamesduan/projects/strategy_agent

# 以root权限运行安装脚本
sudo ./systemd/install.sh
```

安装脚本会自动：
- 复制服务配置文件到 `/etc/systemd/system/`
- 复制日志轮转配置到 `/etc/logrotate.d/`
- 重新加载systemd守护进程
- 启用服务（但不启动）

### 2. 配置环境变量

```bash
# 创建环境变量文件（仅包含敏感信息）
echo 'DEEPSEEK_API_KEY=your_api_key_here' > .env

# 编辑环境变量文件
vim .env
```

必需的环境变量：
- `DEEPSEEK_API_KEY`: DeepSeek API密钥

主要配置项在 `config/development.yaml` 中管理，包括：
- Redis连接配置
- Binance API配置
- 数据收集参数
- 分析器设置
- 日志配置

### 3. 启动服务

```bash
# 启动主服务
sudo systemctl start strategy-agent-data-collector.service

# 启动健康检查定时器
sudo systemctl start strategy-agent-data-collector.timer

# 查看服务状态
sudo systemctl status strategy-agent-data-collector.service
```

## 服务管理命令

### 基本操作

```bash
# 启动服务
sudo systemctl start strategy-agent-data-collector.service

# 停止服务
sudo systemctl stop strategy-agent-data-collector.service

# 重启服务
sudo systemctl restart strategy-agent-data-collector.service

# 重新加载配置
sudo systemctl reload strategy-agent-data-collector.service

# 查看服务状态
sudo systemctl status strategy-agent-data-collector.service
```

### 开机自启

```bash
# 启用开机自启
sudo systemctl enable strategy-agent-data-collector.service

# 禁用开机自启
sudo systemctl disable strategy-agent-data-collector.service

# 查看是否启用
sudo systemctl is-enabled strategy-agent-data-collector.service
```

### 健康检查定时器

```bash
# 启动定时器
sudo systemctl start strategy-agent-data-collector.timer

# 停止定时器
sudo systemctl stop strategy-agent-data-collector.timer

# 查看定时器状态
sudo systemctl status strategy-agent-data-collector.timer

# 查看定时器下次执行时间
systemctl list-timers strategy-agent-data-collector.timer
```

## 日志管理

### 查看日志

```bash
# 查看服务日志（实时）
sudo journalctl -u strategy-agent-data-collector.service -f

# 查看最近的日志
sudo journalctl -u strategy-agent-data-collector.service --since "1 hour ago"

# 查看健康检查日志
sudo journalctl -u strategy-agent-data-collector-health.service -f

# 查看系统日志中的服务相关日志
sudo journalctl -f | grep strategy-agent-data-collector
```

### 日志文件

服务也会将日志写入到项目的 `logs/` 目录：
- `logs/agent_data_collector.log`: 主要日志文件
- `logs/health_check.log`: 健康检查日志

日志轮转配置：
- 每日轮转
- 保留30天历史
- 压缩旧日志
- 最大文件大小100MB

## 故障排除

### 服务无法启动

1. **检查环境变量**：
```bash
# 确认.env文件存在且配置正确
ls -la .env
cat .env
```

2. **检查Python环境**：
```bash
# 确认虚拟环境存在
ls -la venv/bin/python

# 测试手动运行
venv/bin/python agent_data_collector.py
```

3. **检查权限**：
```bash
# 确认文件权限正确
ls -la agent_data_collector.py
ls -la logs/ storage/
```

4. **查看详细错误**：
```bash
# 查看systemd日志
sudo journalctl -u strategy-agent-data-collector.service -n 50
```

### 服务频繁重启

1. **检查资源限制**：
```bash
# 查看内存使用
systemctl status strategy-agent-data-collector.service
free -h

# 调整内存限制（如果需要）
sudo systemctl edit strategy-agent-data-collector.service
# 添加：[Service] MemoryMax=4G
```

2. **检查进程状态**：
```bash
# 查看进程是否运行
pgrep -f agent_data_collector.py

# 查看进程详情
ps aux | grep agent_data_collector
```

### 健康检查失败

1. **查看健康检查日志**：
```bash
sudo journalctl -u strategy-agent-data-collector-health.service -f
```

2. **手动执行健康检查**：
```bash
sudo systemctl start strategy-agent-data-collector-health.service
```

## 监控和告警

### 基础监控

```bash
# 创建监控脚本
cat > monitor_service.sh << 'EOF'
#!/bin/bash
SERVICE="strategy-agent-data-collector.service"
STATUS=$(systemctl is-active $SERVICE)

if [ "$STATUS" != "active" ]; then
    echo "Service $SERVICE is not running (status: $STATUS)"
    # 发送告警邮件或Slack通知
    exit 1
fi
echo "Service $STATUS"
EOF

chmod +x monitor_service.sh
```

### 性能监控

```bash
# 查看资源使用情况
systemctl status strategy-agent-data-collector.service

# 查看详细进程信息
systemctl show strategy-agent-data-collector.service

# 查看cgroup统计
systemd-cgtop
```

## 配置优化

### 调整资源限制

```bash
# 创建systemd drop-in配置
sudo systemctl edit strategy-agent-data-collector.service

# 添加自定义配置
[Service]
MemoryMax=4G
CPUQuota=90%
LimitNOFILE=131072
```

### 调整日志级别

```bash
# 修改.service文件中的日志级别
# Environment=LOG_LEVEL=DEBUG
```

## 卸载

```bash
# 停止并禁用服务
sudo systemctl stop strategy-agent-data-collector.service
sudo systemctl stop strategy-agent-data-collector.timer
sudo systemctl disable strategy-agent-data-collector.service
sudo systemctl disable strategy-agent-data-collector.timer

# 删除配置文件
sudo rm /etc/systemd/system/strategy-agent-data-collector.service
sudo rm /etc/systemd/system/strategy-agent-data-collector-health.service
sudo rm /etc/systemd/system/strategy-agent-data-collector.timer
sudo rm /etc/logrotate.d/strategy-agent-data-collector

# 重新加载systemd
sudo systemctl daemon-reload
```

## 最佳实践

1. **定期检查日志**：设置定时任务检查日志错误
2. **监控资源使用**：关注内存和CPU使用情况
3. **备份配置**：定期备份环境变量和配置文件
4. **测试故障恢复**：定期测试服务重启和故障恢复
5. **版本控制**：将配置文件纳入版本控制（除敏感信息）

## 联系支持

如果遇到问题，请：
1. 查看本文档的故障排除部分
2. 收集相关日志文件
3. 记录错误信息和复现步骤
4. 联系技术支持团队