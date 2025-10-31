#!/usr/bin/env python3
"""精确定时动量策略启动脚本。"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.timed_momentum_strategy import main

if __name__ == "__main__":
    asyncio.run(main())