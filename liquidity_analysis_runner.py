#!/usr/bin/env python3
"""
流动性分析主入口 - 整合数据聚合和AI分析的完整流程
根据历史数据生成分析报告并打印结果
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import argparse

from data_aggregator import DataAggregator
from ai_analysis_client import AIAnalysisClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiquidityAnalysisRunner:
    """流动性分析运行器"""

    def __init__(self, config_path: str = "config/development.yaml"):
        """
        初始化分析运行器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.data_aggregator = DataAggregator(config_path)
        self.ai_client = AIAnalysisClient(config_path)

    async def run_complete_analysis(self, trades_limit: int = 200, save_files: bool = True) -> bool:
        """
        运行完整的分析流程

        Args:
            trades_limit: 交易数据限制条数
            save_files: 是否保存文件

        Returns:
            success: 分析是否成功
        """
        try:
            logger.info("🚀 启动完整的流动性分析流程")
            print("\n" + "="*80)
            print("📊 BTC-FDUSD 流动性密度分析系统")
            print("="*80)

            # 步骤1: 数据获取和聚合
            logger.info("📥 步骤1: 获取和聚合市场数据")
            print("\n🔍 步骤1: 获取市场数据...")

            trades_data = self.data_aggregator.get_trades_window_data(limit=trades_limit)
            depth_data = self.data_aggregator.get_depth_snapshot_data()

            if not trades_data:
                print("❌ 错误: 没有可用的交易数据")
                logger.error("没有可用的交易数据")
                return False

            print(f"✅ 成功获取 {len(trades_data)} 分钟的交易数据")
            print(f"✅ 深度数据: {'已获取' if depth_data else '未获取'}")

            # 步骤2: 数据聚合分析
            logger.info("📊 步骤2: 数据聚合分析")
            print("\n📈 步骤2: 聚合和分析数据...")

            trades_aggregated = self.data_aggregator.aggregate_trades_data(trades_data)
            depth_aggregated = self.data_aggregator.aggregate_depth_data(depth_data)

            # 打印数据摘要
            self._print_data_summary(trades_aggregated, depth_aggregated)

            # 步骤3: 格式化数据
            logger.info("📝 步骤3: 格式化数据为AI可读格式")
            print("\n📝 步骤3: 格式化数据...")

            formatted_data = self.data_aggregator.format_for_ai_analysis(trades_aggregated, depth_aggregated)

            if save_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_filename = f"market_data_{timestamp}.txt"
                self.data_aggregator.save_formatted_data(formatted_data, data_filename)
                print(f"✅ 格式化数据已保存到: {data_filename}")

            # 步骤4: AI分析
            logger.info("🤖 步骤4: AI智能分析")
            print("\n🤖 步骤4: DeepSeek AI 分析中...")
            print("⏳ 正在调用AI进行流动性密度分析，请稍候...")

            analysis_result = await self.ai_client.analyze_market_data(trades_limit=trades_limit)

            if analysis_result:
                logger.info("✅ AI分析完成")
                print("\n✅ AI分析完成!")

                # 步骤5: 显示结果
                self._print_analysis_result(analysis_result)

                return True
            else:
                print("❌ AI分析失败")
                logger.error("AI分析失败")
                return False

        except Exception as e:
            error_msg = f"分析流程异常: {e}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            return False

    def _print_data_summary(self, trades_aggregated: dict, depth_aggregated: dict):
        """打印数据摘要"""
        print("\n📊 数据摘要:")

        if trades_aggregated:
            summary = trades_aggregated.get('summary', {})
            time_analysis = trades_aggregated.get('time_analysis', {})
            price_analysis = trades_aggregated.get('price_analysis', {})

            print(f"  ⏰ 时间范围: {time_analysis.get('duration_hours', 0):.1f} 小时")
            print(f"  💰 总成交量: {summary.get('total_volume_btc', 0)} BTC")
            print(f"  📈 总交易次数: {summary.get('total_trades', 0)}")
            print(f"  🐂 买盘比例: {summary.get('buy_ratio', 0):.1%}")
            print(f"  🐻 卖盘比例: {summary.get('sell_ratio', 0):.1%}")
            print(f"  💵 当前价格: ${summary.get('current_price', 0):,.2f}")
            print(f"  📊 价格区间: ${price_analysis.get('min_price', 0):,.2f} - ${price_analysis.get('max_price', 0):,.2f}")
            print(f"  🎯 价格水平数: {price_analysis.get('price_levels_analyzed', 0)}")

        if depth_aggregated:
            spread_analysis = depth_aggregated.get('spread_analysis', {})
            market_depth = depth_aggregated.get('market_depth', {})

            print(f"  📚 订单簿价差: ${spread_analysis.get('spread', 0):,.2f} ({spread_analysis.get('spread_percentage', 0):.4f}%)")
            print(f"  🎯 中间价格: ${market_depth.get('mid_price', 0):,.2f}")
            print(f"  💎 总深度价值: ${market_depth.get('total_depth_value', 0):,.2f}")

    def _print_analysis_result(self, result: str):
        """打印分析结果"""
        print("\n" + "="*80)
        print("🤖 DEEPSEEK AI 流动性密度分析结果")
        print("="*80)

        # 显示完整结果
        print(result)

        print("\n" + "="*80)
        print("📋 分析总结")
        print("="*80)

        # 尝试提取关键信息并显示简化总结
        self._extract_and_display_summary(result)

        print("\n✅ 分析完成!")
        print("📄 详细分析结果和生成的文件已保存到当前目录")
        print("🎯 建议基于AI分析进行交易决策时严格遵守风险管理原则")

    def _extract_and_display_summary(self, result: str):
        """提取并显示分析结果摘要"""
        try:
            lines = result.split('\n')
            summary_lines = []

            # 查找关键信息
            key_indicators = [
                "Primary Liquidity Zone",
                "Center Price:",
                "Price Range:",
                "Optimal Entry Price:",
                "Stop Loss:",
                "Take Profit:",
                "Risk/Reward Ratio:",
                "Win Rate Probability:",
                "Trend Direction:",
                "Strong Support:",
                "Overall Recommendation:"
            ]

            for line in lines:
                for indicator in key_indicators:
                    if indicator in line:
                        summary_lines.append(line.strip())
                        break

            if summary_lines:
                print("🎯 关键信息提取:")
                for line in summary_lines[:15]:  # 显示前15行关键信息
                    print(f"  {line}")

                if len(summary_lines) > 15:
                    print(f"  ... 还有 {len(summary_lines) - 15} 行关键信息")
            else:
                print("📝 无法提取关键信息，请查看完整分析结果")

        except Exception as e:
            logger.warning(f"提取摘要失败: {e}")
            print("📝 摘要提取失败，请查看完整分析结果")

    async def run_data_only(self, trades_limit: int = 200):
        """仅运行数据聚合，不进行AI分析"""
        try:
            logger.info("📊 仅运行数据聚合分析")
            print("\n" + "="*60)
            print("📊 数据聚合模式")
            print("="*60)

            # 获取数据
            trades_data = self.data_aggregator.get_trades_window_data(limit=trades_limit)
            depth_data = self.data_aggregator.get_depth_snapshot_data()

            if not trades_data:
                print("❌ 没有可用的交易数据")
                return

            print(f"✅ 获取 {len(trades_data)} 分钟交易数据")

            # 聚合数据
            trades_aggregated = self.data_aggregator.aggregate_trades_data(trades_data)
            depth_aggregated = self.data_aggregator.aggregate_depth_data(depth_data)

            # 显示摘要
            self._print_data_summary(trades_aggregated, depth_aggregated)

            # 格式化并保存
            formatted_data = self.data_aggregator.format_for_ai_analysis(trades_aggregated, depth_aggregated)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_data_{timestamp}.txt"
            self.data_aggregator.save_formatted_data(formatted_data, filename)

            print(f"\n✅ 数据聚合完成!")
            print(f"📄 格式化数据已保存到: {filename}")
            print("💡 使用 --ai 参数进行AI分析")

        except Exception as e:
            logger.error(f"数据聚合失败: {e}")
            print(f"❌ 数据聚合失败: {e}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BTC-FDUSD 流动性密度分析系统')
    parser.add_argument('--config', default='config/development.yaml', help='配置文件路径')
    parser.add_argument('--trades-limit', type=int, default=200, help='交易数据限制条数')
    parser.add_argument('--no-save', action='store_true', help='不保存文件到磁盘')
    parser.add_argument('--data-only', action='store_true', help='仅进行数据聚合，不进行AI分析')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 创建分析运行器
        runner = LiquidityAnalysisRunner(args.config)

        if args.data_only:
            # 仅数据聚合模式
            await runner.run_data_only(trades_limit=args.trades_limit)
        else:
            # 完整分析模式
            success = await runner.run_complete_analysis(
                trades_limit=args.trades_limit,
                save_files=not args.no_save
            )

            if success:
                print("\n🎉 流动性密度分析成功完成!")
            else:
                print("\n❌ 分析失败，请检查配置和数据")
                exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断分析")
        logger.info("用户中断分析")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        logger.error(f"程序异常: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())