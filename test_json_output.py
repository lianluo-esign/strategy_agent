#!/usr/bin/env python3
"""
测试JSON输出格式 - 模拟DeepSeek返回的JSON响应
"""

import json
from enhanced_ai_client import EnhancedAIClient

def test_json_parsing():
    """测试JSON解析功能"""
    print("🧪 测试JSON输出格式解析")
    print("="*50)

    # 创建AI客户端实例
    client = EnhancedAIClient()

    # 测试用例
    test_cases = [
        # 标准Buy信号
        {
            "name": "标准做多信号",
            "json_str": """{
  "direction": "Buy",
  "lower_bound": 114650.0,
  "upper_bound": 115800.0
}""",
            "expected_direction": "Buy",
            "expected_lower": 114650.0,
            "expected_upper": 115800.0
        },
        # 标准Sell信号
        {
            "name": "标准做空信号",
            "json_str": """{
  "direction": "Sell",
  "lower_bound": 113800.0,
  "upper_bound": 115200.0
}""",
            "expected_direction": "Sell",
            "expected_lower": 113800.0,
            "expected_upper": 115200.0
        },
        # 标准Hold信号
        {
            "name": "标准持有信号",
            "json_str": """{
  "direction": "Hold",
  "lower_bound": 114500.0,
  "upper_bound": 115500.0
}""",
            "expected_direction": "Hold",
            "expected_lower": 114500.0,
            "expected_upper": 115500.0
        },
        # 带额外文本的JSON
        {
            "name": "带额外文本的JSON",
            "json_str": """分析完成，建议如下：
{
  "direction": "Buy",
  "lower_bound": 114700.0,
  "upper_bound": 116000.0
}
基于当前市场状况...""",
            "expected_direction": "Buy",
            "expected_lower": 114700.0,
            "expected_upper": 116000.0
        },
        # 无效JSON（回退测试）
        {
            "name": "无效JSON回退测试",
            "json_str": "基于分析，建议BUY在114000-116000区间",
            "expected_direction": "Buy",  # 通过关键词匹配
            "expected_lower": None,
            "expected_upper": None
        }
    ]

    # 执行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 测试 {i}: {test_case['name']}")
        print("-" * 30)

        # 解析结果
        result = client.extract_decision_summary(test_case['json_str'])

        # 显示原始输入
        print(f"输入: {test_case['json_str'][:100]}...")

        # 显示解析结果
        print(f"解析结果:")
        print(f"  方向: {result.get('direction', 'N/A')}")
        print(f"  下界: {result.get('lower_bound', 'N/A')}")
        print(f"  上界: {result.get('upper_bound', 'N/A')}")
        print(f"  有效: {result.get('valid', 'N/A')}")

        # 验证期望结果
        if result.get('direction') == test_case['expected_direction']:
            print("  ✅ 方向匹配")
        else:
            print(f"  ❌ 方向不匹配 (期望: {test_case['expected_direction']})")

        if test_case['expected_lower'] is not None:
            if result.get('lower_bound') == test_case['expected_lower']:
                print("  ✅ 下界匹配")
            else:
                print(f"  ❌ 下界不匹配 (期望: {test_case['expected_lower']})")

        if test_case['expected_upper'] is not None:
            if result.get('upper_bound') == test_case['expected_upper']:
                print("  ✅ 上界匹配")
            else:
                print(f"  ❌ 上界不匹配 (期望: {test_case['expected_upper']})")

    print("\n" + "="*50)
    print("✅ JSON解析测试完成")

def simulate_json_display():
    """模拟JSON格式显示"""
    print("\n🎯 模拟JSON输出显示效果")
    print("="*50)

    # 模拟不同的JSON响应
    sample_responses = [
        {
            "title": "做多信号示例",
            "json": {
                "direction": "Buy",
                "lower_bound": 114650.0,
                "upper_bound": 115800.0
            }
        },
        {
            "title": "做空信号示例",
            "json": {
                "direction": "Sell",
                "lower_bound": 113800.0,
                "upper_bound": 115200.0
            }
        },
        {
            "title": "持有信号示例",
            "json": {
                "direction": "Hold",
                "lower_bound": 114500.0,
                "upper_bound": 115500.0
            }
        }
    ]

    client = EnhancedAIClient()

    for sample in sample_responses:
        print(f"\n📊 {sample['title']}")
        print("-" * 30)

        # 模拟AI返回JSON
        json_str = json.dumps(sample['json'], indent=2)
        print("🤖 AI返回:")
        print(json_str)

        # 解析并显示
        result = client.extract_decision_summary(json_str)
        print(f"\n📋 解析结果:")
        if result.get('valid', False):
            direction_emoji = {
                'Buy': '📈 做多',
                'Sell': '📉 做空',
                'Hold': '⏸️ 持有'
            }.get(result.get('direction'), f'❓ {result.get("direction")}')

            print(f"🎯 推荐操作: {direction_emoji}")
            print(f"📊 交易区间: ${result.get('lower_bound', 0):,.2f} - ${result.get('upper_bound', 0):,.2f}")
            print(f"📏 区间宽度: ${result.get('trading_range', 0):,.2f}")
            print(f"✅ 数据格式: JSON (有效)")

            # 提供执行建议
            if result.get('direction') == 'Buy':
                print(f"💡 建议在区间下限 ${result.get('lower_bound', 0):,.2f} 附近入场")
                print(f"🎯 目标价位区间上限 ${result.get('upper_bound', 0):,.2f}")
            elif result.get('direction') == 'Sell':
                print(f"💡 建议在区间上限 ${result.get('upper_bound', 0):,.2f} 附近入场")
                print(f"🎯 目标价位区间下限 ${result.get('lower_bound', 0):,.2f}")
            else:  # Hold
                print(f"💡 建议在区间 ${result.get('lower_bound', 0):,.2f} - ${result.get('upper_bound', 0):,.2f} 内观察")
        else:
            print(f"❌ 数据解析失败")

if __name__ == "__main__":
    test_json_parsing()
    simulate_json_display()