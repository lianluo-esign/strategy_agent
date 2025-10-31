#!/bin/bash

# 流动性密度分析系统快速启动脚本

set -e

echo "🚀 BTC-FDUSD 流动性密度分析系统"
echo "=================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "⚠️  警告: 未找到虚拟环境，建议使用venv"
    echo "💡 创建虚拟环境: python3 -m venv venv && source venv/bin/activate"
fi

# 检查配置文件
if [ ! -f "config/development.yaml" ]; then
    echo "❌ 错误: 未找到配置文件 config/development.yaml"
    exit 1
fi

# 检查Redis连接
echo "🔍 检查Redis连接..."
if python3 -c "
import redis
import yaml
import sys
try:
    with open('config/development.yaml') as f:
        config = yaml.safe_load(f)
    redis_config = config.get('redis', {})
    r = redis.Redis(
        host=redis_config.get('host', 'localhost'),
        port=redis_config.get('port', 6379),
        db=redis_config.get('db', 0),
        decode_responses=True,
        socket_timeout=5
    )
    r.ping()
    print('✅ Redis连接成功')
    sys.exit(0)
except Exception as e:
    print(f'❌ Redis连接失败: {e}')
    print('💡 请确保Redis服务正在运行')
    sys.exit(1)
"; then
    echo "✅ Redis连接正常"
else
    echo "❌ Redis连接失败，请检查Redis服务"
    exit 1
fi

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo "📦 激活虚拟环境..."
    source venv/bin/activate
fi

# 获取命令行参数
TRADING_LIMIT=${1:-200}
DATA_ONLY=${2:-false}

echo "📊 配置参数:"
echo "  交易数据限制: ${TRADING_LIMIT} 分钟"
echo "  仅数据模式: ${DATA_ONLY}"

# 运行分析
if [ "$DATA_ONLY" = "true" ]; then
    echo "🔍 运行数据聚合分析..."
    python3 liquidity_analysis_runner.py --data-only --trades-limit "$TRADING_LIMIT"
else
    echo "🤖 运行完整AI分析..."
    python3 liquidity_analysis_runner.py --trades-limit "$TRADING_LIMIT"
fi

echo ""
echo "✅ 分析完成!"
echo "📄 生成的文件:"
if [ -f "market_data_"*.txt ]; then
    ls -la market_data_*.txt | tail -1
fi
if [ -f "ai_prompt_"*.txt ]; then
    ls -la ai_prompt_*.txt | tail -1
fi
if [ -f "ai_analysis_"*.txt ]; then
    ls -la ai_analysis_*.txt | tail -1
fi

echo ""
echo "💡 使用帮助:"
echo "  ./run_analysis.sh [数据量] [仅数据模式]"
echo "  例如: ./run_analysis.sh 150 false"
echo "  例如: ./run_analysis.sh 100 true"
echo ""
echo "📖 详细说明请查看 README_liquidity_analysis.md"