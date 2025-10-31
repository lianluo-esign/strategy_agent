#!/usr/bin/env python3
"""测试短期动量策略启动器的脚本。"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.models import Trade
from momentum_strategy import MomentumStrategyLauncher


class TestMomentumStrategyLauncher(MomentumStrategyLauncher):
    """测试用的短期动量策略启动器。"""

    async def _generate_mock_trades(self) -> list[Trade]:
        """生成更好的模拟交易数据。"""
        trades = []
        current_time = datetime.now()
        base_price = Decimal("42000.00")

        # 模拟不同的市场场景
        scenarios = ["uptrend", "downtrend", "sideways"]
        scenario = scenarios[self.signal_count % len(scenarios)]

        trades_per_batch = 20  # 增加交易数量

        for i in range(trades_per_batch):
            # 确保交易在5分钟窗口内（更密集的时间分布）
            timestamp = current_time - timedelta(seconds=i * 12)  # 每12秒一笔交易

            if scenario == "uptrend":
                # 上升趋势
                price = base_price + Decimal(str(i * 0.8))
                is_buyer_maker = i % 4 != 0  # 更多买入
            elif scenario == "downtrend":
                # 下跌趋势
                price = base_price - Decimal(str(i * 0.6))
                is_buyer_maker = i % 3 == 0  # 更多卖出
            else:
                # 横盘整理
                price = base_price + Decimal(str((i % 10 - 5) * 0.2))
                is_buyer_maker = i % 2 == 0

            # 模拟交易量
            quantity = Decimal(str(0.1 + (i % 8) * 0.02))

            trade = Trade(
                symbol=self.config["data"]["symbol"],
                price=price,
                quantity=quantity,
                is_buyer_maker=is_buyer_maker,
                timestamp=timestamp,
                trade_id=f"test_trade_{int(current_time.timestamp())}_{i}"
            )
            trades.append(trade)

        print(f"🎭 测试场景: {scenario} - 生成了 {len(trades)} 笔交易")
        return trades


async def test_strategy():
    """测试策略运行。"""
    print("🧪 开始测试短期动量策略启动器")
    print("=" * 60)

    # 创建测试启动器
    launcher = TestMomentumStrategyLauncher("config/momentum_strategy.yaml")

    # 修改配置以便快速测试
    launcher.config["analysis_interval_seconds"] = 3

    try:
        # 运行3个周期后停止
        original_run = launcher.run_analysis_loop

        async def limited_run():
            launcher.is_running = True
            launcher.start_time = datetime.now()

            for i in range(3):
                if not launcher.is_running:
                    break

                cycle_start = asyncio.get_event_loop().time()

                # 获取交易数据
                trades_data = await launcher._get_trades_data()

                if trades_data:
                    # 执行动量分析
                    result = launcher.analyzer.analyze_momentum(
                        trades_data,
                        launcher.config["data"]["symbol"]
                    )

                    # 输出信号
                    launcher._output_signal(result)
                    launcher.signal_count += 1

                # 等待间隔时间
                if i < 2:  # 最后一次不需要等待
                    await asyncio.sleep(launcher.config["analysis_interval_seconds"])

        await limited_run()

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise
    finally:
        await launcher._shutdown()

    print("✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(test_strategy())