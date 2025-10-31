#!/usr/bin/env python3
"""
交易决策主入口 - 整合精细化数据聚合和AI分析
专门用于做多/做空/持有决策分析
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional
import argparse

from enhanced_data_aggregator import EnhancedDataAggregator
from enhanced_ai_client import EnhancedAIClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingDecisionRunner:
    """交易决策运行器"""

    def __init__(self, config_path: str = "config/development.yaml"):
        """
        初始化分析运行器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.data_aggregator = EnhancedDataAggregator(config_path)
        self.ai_client = EnhancedAIClient(config_path)

    async def run_complete_analysis(self, save_files: bool = True) -> bool:
        """
        运行完整的交易决策分析流程

        Args:
            save_files: 是否保存文件

        Returns:
            success: 分析是否成功
        """
        try:
            logger.info("🚀 启动交易决策分析流程")
            print("\n" + "="*80)
            print("📊 BTC-FDUSD 交易决策分析系统 (做多/做空/持有)")
            print("="*80)

            # 步骤1: 获取和处理深度数据
            logger.info("📚 步骤1: 获取和聚合深度数据 (10美元精度)")
            print("\n📚 步骤1: 深度订单簿聚合...")

            depth_str = self.data_aggregator.redis.get('depth_snapshot_5000')
            if not depth_str:
                print("❌ 错误: 没有可用的深度数据")
                logger.error("没有可用的深度数据")
                return False

            depth_data = json.loads(depth_str)
            aggregated_depth = self.data_aggregator.aggregate_depth_data(depth_data)

            self._print_depth_summary(aggregated_depth)

            # 步骤2: 获取交易数据
            logger.info("📈 步骤2: 获取多周期交易数据")
            print("\n📈 步骤2: 获取交易数据...")

            recent_trades = self.data_aggregator.get_recent_trades_data(minutes=30)
            if not recent_trades:
                print("❌ 错误: 没有可用的最近交易数据")
                logger.error("没有可用的最近交易数据")
                return False

            print(f"✅ 获取最近30分钟数据: {len(recent_trades)} 分钟")

            all_trades = self.data_aggregator.get_recent_trades_data(minutes=2880)
            aggregated_4h = self.data_aggregator.aggregate_4h_data(all_trades)

            print(f"✅ 获取4小时聚合数据: {len(aggregated_4h)} 个时间段")

            self._print_trades_summary(recent_trades, aggregated_4h)

            # 步骤3: 格式化数据
            logger.info("📝 步骤3: 格式化数据为AI可读格式")
            print("\n📝 步骤3: 格式化分析数据...")

            formatted_data = self.data_aggregator.format_data_for_ai(
                recent_trades, aggregated_depth, aggregated_4h
            )

            if save_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_filename = f"trading_analysis_data_{timestamp}.txt"
                self.data_aggregator.save_formatted_data(formatted_data, data_filename)
                print(f"✅ 格式化数据已保存到: {data_filename}")

            # 步骤4: AI交易决策分析
            logger.info("🤖 步骤4: AI交易决策分析")
            print("\n🤖 步骤4: DeepSeek AI 交易决策分析中...")
            print("⏳ 正在分析做多/做空/持有决策，请稍候...")

            # 生成提示词
            prompt = self.ai_client.generate_trading_decision_prompt(formatted_data)

            # 调用AI API
            analysis_result = await self.ai_client.call_deepseek_api(prompt)

            if analysis_result:
                logger.info("✅ AI分析完成")
                print("\n✅ AI交易决策分析完成!")

                # 保存提示词和分析结果
                if save_files:
                    prompt_filename = f"trading_decision_prompt_{timestamp}.txt"
                    result_filename = f"trading_decision_analysis_{timestamp}.txt"

                    with open(prompt_filename, 'w', encoding='utf-8') as f:
                        f.write(prompt)
                    with open(result_filename, 'w', encoding='utf-8') as f:
                        f.write(analysis_result)

                    print(f"✅ 提示词已保存到: {prompt_filename}")
                    print(f"✅ 分析结果已保存到: {result_filename}")

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

    def _print_depth_summary(self, aggregated_depth: dict):
        """打印深度数据摘要"""
        print("\n📊 深度数据摘要:")

        if not aggregated_depth:
            print("  ❌ 没有深度数据")
            return

        market_summary = aggregated_depth.get('market_summary', {})
        price_levels = aggregated_depth.get('price_levels', {})

        print(f"  📈 最佳买价: ${market_summary.get('best_bid', 0):,.2f}")
        print(f"  📉 最佳卖价: ${market_summary.get('best_ask', 0):,.2f}")
        print(f"  📏 价差: ${market_summary.get('spread', 0):,.2f} ({market_summary.get('spread_percentage', 0):.4f}%)")
        print(f"  📊 总价格档位: {price_levels.get('total_levels', 0)}")
        print(f"  📉 卖方档位: {price_levels.get('ask_levels', 0)}")
        print(f"  📈 买方档位: {price_levels.get('bid_levels', 0)}")

        ask_stats = market_summary.get('ask_stats', {})
        bid_stats = market_summary.get('bid_stats', {})

        print(f"  💰 卖方总挂单: {ask_stats.get('total_quantity', 0):.4f} BTC")
        print(f"  💰 买方总挂单: {bid_stats.get('total_quantity', 0):.4f} BTC")
        print(f"  🎯 卖方集中度(前3档): {ask_stats.get('concentration', {}).get('top_3_levels_ratio', 0):.1%}")
        print(f"  🎯 买方集中度(前3档): {bid_stats.get('concentration', {}).get('top_3_levels_ratio', 0):.1%}")

    def _print_trades_summary(self, recent_trades: list, aggregated_4h: list):
        """打印交易数据摘要"""
        print("\n📊 交易数据摘要:")

        # 最近30分钟数据摘要
        total_volume = 0
        total_trades = 0
        buy_volume = 0

        for minute_data in recent_trades:
            price_levels = minute_data.get('price_levels', {})
            for level_data in price_levels.values():
                total_volume += level_data.get('total_volume', 0)
                total_trades += level_data.get('trade_count', 0)
                buy_volume += level_data.get('buy_volume', 0)

        print(f"  ⏰ 最近30分钟:")
        print(f"    💰 总成交量: {total_volume:.4f} BTC")
        print(f"    📈 总交易次数: {total_trades}")
        print(f"    🐂 买盘占比: {buy_volume/total_volume:.1%}" if total_volume > 0 else "    🐂 买盘占比: N/A")

        # 4小时数据摘要
        if aggregated_4h:
            latest_period = aggregated_4h[-1]  # 最新的4小时段
            price_analysis = latest_period.get('price_analysis', {})
            volume_analysis = latest_period.get('volume_analysis', {})

            print(f"  📊 最新4小时段:")
            print(f"    💹 价格区间: ${price_analysis.get('low', 0):,.2f} - ${price_analysis.get('high', 0):,.2f}")
            print(f"    📈 价格变化: {price_analysis.get('price_change', 0):+.2%}")
            print(f"    💰 成交量: {volume_analysis.get('total_volume', 0):.4f} BTC")
            print(f"    🐂 买盘占比: {volume_analysis.get('buy_ratio', 0):.1%}")

    def _print_analysis_result(self, result: str):
        """打印分析结果"""
        print("\n" + "="*80)
        print("🤖 DEEPSEEK AI 交易决策分析结果")
        print("="*80)

        # 显示完整结果
        print(result)

        # 提取并显示决策摘要
        summary = self.ai_client.extract_decision_summary(result)

        print("\n" + "="*80)
        print("📋 决策摘要")
        print("="*80)

        # 显示JSON格式的决策结果
        if summary.get('valid', False):
            direction_emoji = {
                'Buy': '📈 做多',
                'Sell': '📉 做空',
                'Hold': '⏸️ 持有'
            }.get(summary.get('direction'), f'❓ {summary.get("direction")}')

            print(f"🎯 推荐操作: {direction_emoji}")
            print(f"📊 交易区间: ${summary.get('lower_bound', 0):,.2f} - ${summary.get('upper_bound', 0):,.2f}")
            print(f"📏 区间宽度: ${summary.get('trading_range', 0):,.2f}")

            if 'direction_cn' in summary:
                print(f"🌐 中文方向: {summary['direction_cn']}")

            print(f"✅ 数据格式: JSON (有效)")

            # 提供交易执行建议
            if summary.get('direction') == 'Buy':
                print(f"💡 建议在区间下限 ${summary.get('lower_bound', 0):,.2f} 附近入场")
                print(f"🎯 目标价位区间上限 ${summary.get('upper_bound', 0):,.2f}")
            elif summary.get('direction') == 'Sell':
                print(f"💡 建议在区间上限 ${summary.get('upper_bound', 0):,.2f} 附近入场")
                print(f"🎯 目标价位区间下限 ${summary.get('lower_bound', 0):,.2f}")
            else:  # Hold
                print(f"💡 建议在区间 ${summary.get('lower_bound', 0):,.2f} - ${summary.get('upper_bound', 0):,.2f} 内观察")
                print(f"🎯 突破区间再考虑入场")

        else:
            print(f"🎯 推荐操作: {summary.get('direction', 'UNKNOWN')}")
            print(f"❌ 数据解析失败，请检查原始结果")
            if 'error' in summary:
                print(f"🔍 错误信息: {summary['error']}")

        print("\n✅ 交易决策分析完成!")
        print("📄 详细分析结果和生成的文件已保存到当前目录")
        print("🎯 请基于AI建议结合自身风险承受能力做出最终决策")
        print("📋 分析格式: JSON标准化输出，便于程序化处理")

    async def run_data_only(self):
        """仅运行数据聚合，不进行AI分析"""
        try:
            logger.info("📊 仅运行数据聚合")
            print("\n" + "="*60)
            print("📊 数据聚合模式")
            print("="*60)

            # 获取深度数据
            depth_str = self.data_aggregator.redis.get('depth_snapshot_5000')
            if not depth_str:
                print("❌ 没有深度数据")
                return

            depth_data = json.loads(depth_str)
            aggregated_depth = self.data_aggregator.aggregate_depth_data(depth_data)

            # 获取交易数据
            recent_trades = self.data_aggregator.get_recent_trades_data(minutes=30)
            all_trades = self.data_aggregator.get_recent_trades_data(minutes=2880)
            aggregated_4h = self.data_aggregator.aggregate_4h_data(all_trades)

            # 显示摘要
            self._print_depth_summary(aggregated_depth)
            self._print_trades_summary(recent_trades, aggregated_4h)

            # 格式化并保存
            formatted_data = self.data_aggregator.format_data_for_ai(
                recent_trades, aggregated_depth, aggregated_4h
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_analysis_data_{timestamp}.txt"
            self.data_aggregator.save_formatted_data(formatted_data, filename)

            print(f"\n✅ 数据聚合完成!")
            print(f"📄 格式化数据已保存到: {filename}")
            print("💡 使用 --ai 参数进行AI交易决策分析")

        except Exception as e:
            logger.error(f"数据聚合失败: {e}")
            print(f"❌ 数据聚合失败: {e}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BTC-FDUSD 交易决策分析系统')
    parser.add_argument('--config', default='config/development.yaml', help='配置文件路径')
    parser.add_argument('--no-save', action='store_true', help='不保存文件到磁盘')
    parser.add_argument('--data-only', action='store_true', help='仅进行数据聚合，不进行AI分析')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 创建分析运行器
        runner = TradingDecisionRunner(args.config)

        if args.data_only:
            # 仅数据聚合模式
            await runner.run_data_only()
        else:
            # 完整分析模式
            success = await runner.run_complete_analysis(save_files=not args.no_save)

            if success:
                print("\n🎉 交易决策分析成功完成!")
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