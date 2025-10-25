#!/usr/bin/env python3
"""简化分析器启动脚本。

这个脚本启动简化的市场分析器，专注于核心功能：
1. Redis深度快照数据读取和聚合
2. DeepSeek LLM支撑阻力分析
3. 挂单分布可视化
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.simple_analyzer_agent import main


if __name__ == "__main__":
    print("🚀 Starting Simple Market Analyzer...")
    print("📊 Core Features:")
    print("   - Redis depth snapshot data reading and aggregation")
    print("   - DeepSeek LLM support/resistance analysis")
    print("   - Order book distribution visualization")
    print("=" * 50)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Simple Analyzer stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)