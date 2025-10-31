# 🤖 DeepSeek AI JSON输出格式示例

## 标准JSON输出格式

AI分析完成后，DeepSeek会返回标准JSON格式：

```json
{
  "direction": "Buy",
  "lower_bound": 114650.0,
  "upper_bound": 115800.0
}
```

## 输出字段说明

### direction (string)
**必需字段**，可能的值：
- `"Buy"` - 做多建议
- `"Sell"` - 做空建议
- `"Hold"` - 持有建议

### lower_bound (number)
**必需字段**，交易区间下边界：
- 对于"Buy"：建议的支撑位或入场价格
- 对于"Sell"：建议的目标位或支撑位
- 对于"Hold"：当前价格区间的下限

### upper_bound (number)
**必需字段**，交易区间上边界：
- 对于"Buy"：建议的目标位或阻力位
- 对于"Sell"：建议的阻力位或入场价格
- 对于"Hold"：当前价格区间的上限

## 系统解析逻辑

### 1. JSON验证
系统会验证返回的JSON是否包含所有必需字段
如果缺少字段，会尝试回退提取方法

### 2. 逻辑验证
确保 `lower_bound < upper_bound`
如果逻辑错误，会标记为无效数据

### 3. 数据增强
系统会自动计算：
- `trading_range`: 区间宽度 (upper_bound - lower_bound)
- `direction_cn`: 中文方向显示
- `valid`: 数据是否有效

## 示例场景

### 做多场景
```json
{
  "direction": "Buy",
  "lower_bound": 114650.0,
  "upper_bound": 115800.0
}
```
**解析结果**：
- 建议在 $114,650 附近入场做多
- 目标价位 $115,800
- 区间宽度：$1,150

### 做空场景
```json
{
  "direction": "Sell",
  "lower_bound": 113800.0,
  "upper_bound": 115200.0
}
```
**解析结果**：
- 建议在 $115,200 附近入场做空
- 目标价位 $113,800
- 区间宽度：$1,400

### 持有场景
```json
{
  "direction": "Hold",
  "lower_bound": 114500.0,
  "upper_bound": 115500.0
}
```
**解析结果**：
- 建议在 $114,500 - $115,500 区间内观察
- 突破区间后再考虑入场
- 当前处于震荡阶段

## 显示格式

系统会以用户友好的格式显示解析结果：

```
📋 决策摘要
============================
🎯 推荐操作: 📈 做多
📊 交易区间: $114,650.00 - $115,800.00
📏 区间宽度: $1,150.00
🌐 中文方向: 做多
✅ 数据格式: JSON (有效)

💡 建议在区间下限 $114,650.00 附近入场
🎯 目标价位区间上限 $115,800.00

✅ 交易决策分析完成!
📋 分析格式: JSON标准化输出，便于程序化处理
```

## 错误处理

### JSON解析失败
如果JSON格式错误，系统会：
1. 尝试从文本中提取JSON片段
2. 回退到关键词匹配
3. 标记数据为无效

### 逻辑错误
如果 `lower_bound >= upper_bound`，系统会：
1. 记录错误日志
2. 标记数据为无效
3. 显示解析失败警告

### 字段缺失
如果缺少必需字段，系统会：
1. 使用回退提取方法
2. 设置默认值
3. 标记数据完整性警告

## 技术实现

### JSON解析代码示例
```python
def extract_decision_summary(self, result: str) -> Dict:
    try:
        decision_data = json.loads(result.strip())

        # 验证必需字段
        if not all(key in decision_data for key in ['direction', 'lower_bound', 'upper_bound']):
            return self._fallback_extraction(result)

        # 数据增强
        summary = {
            'direction': decision_data['direction'],
            'lower_bound': decision_data['lower_bound'],
            'upper_bound': decision_data['upper_bound'],
            'trading_range': decision_data['upper_bound'] - decision_data['lower_bound'],
            'valid': decision_data['lower_bound'] < decision_data['upper_bound']
        }

        return summary

    except json.JSONDecodeError:
        return self._fallback_extraction(result)
```

### 提示词关键部分
```text
You MUST respond with a valid JSON object ONLY. No other text, explanations, or formatting outside the JSON structure.

Your JSON response must have exactly these properties:
- direction: "Buy", "Sell", or "Hold"
- lower_bound: numeric value (float or integer) representing the lower bound of the trading range
- upper_bound: numeric value (float or integer) representing the upper bound of the trading range
```

---

**更新时间**: 2025-10-28
**版本**: 3.0.0
**输出格式**: 标准JSON