# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个专为交易所BTC-FDUSD现货交易设计的加密货币流动性分析agent。该系统通过分析静态订单簿深度和动态订单流数据来识别高概率交易机会，实现复杂的做市策略。

## 核心交易逻辑：从"静态阻力"到"动态战场"

系统通过三个agent来分析市场结构：

### 静态支撑/阻力分析
- **订单簿墙**：在5000级深度快照中识别重要的订单集中区域
- **高密度区**：映射具有持续订单量的连续价格区间
- **流动性真空区**：检测容易出现快速价格滑点的订单簿稀薄区域

### 通过订单流进行动态确认
- **控制点(POC)**：分析3天订单流数据以找到共识价格水平
- **有效支撑/阻力**：通过实际市场吸收/拒绝模式确认静态水平
- **市场影响分析**：跟踪大订单在关键位置的吸收或拒绝情况

### 综合分析与策略执行
- **共振区**：识别静态和动态分析收敛的区域
- **最优摆放**：将流动性放置在大型订单前方以获得更好的执行
- **仓位规模**：根据区域强度和确认频率分配资本

## Python开发项目约束

### 代码编写规范

#### 1. 代码风格
- **强制使用**：`ruff` 进行代码格式化和检查
- **类型注解**：所有函数必须包含完整的类型注解
- **文档字符串**：所有公共函数和类必须包含详细的docstring
- **变量命名**：使用有意义的英文变量名，避免缩写
- **常量定义**：使用大写字母定义常量，集中放在`constants.py`文件中

#### 2. 代码结构
- **单一职责**：每个函数只负责一个明确的功能
- **函数长度**：单个函数不超过50行
- **类设计**：保持类的简单性和内聚性
- **依赖注入**：使用依赖注入来提高可测试性
- **异常处理**：使用具体的异常类型，避免裸露的except

#### 3. 性能要求
- **内存管理**：及时释放不需要的对象，避免内存泄漏
- **异步编程**：I/O密集型操作必须使用async/await
- **数据结构选择**：根据场景选择合适的数据结构
- **缓存策略**：对计算密集型操作实现适当的缓存

### 单元测试要求

#### 1. 测试覆盖率
- **最低覆盖率**：90%的代码覆盖率
- **分支覆盖**：确保所有条件分支都被测试
- **边界测试**：测试所有边界条件和极端情况
- **异常路径**：测试所有异常处理路径

#### 2. 测试结构
```python
# 标准测试文件结构
class TestClassName:
    def setup_method(self):
        """每个测试方法前的设置"""
        pass

    def teardown_method(self):
        """每个测试方法后的清理"""
        pass

    def test_method_should_return_expected_result(self):
        """测试方法应该返回预期结果"""
        # Arrange
        # Act
        # Assert
        pass
```

#### 3. 测试命名约定
- **描述性命名**：测试名称应该清楚描述测试的内容
- **Given_When_Then**：使用BDD风格的测试命名
- **测试场景**：每个测试只验证一个场景

#### 4. Mock和Fixture使用
- **合理Mock**：只mock外部依赖，避免过度mock
- **Fixture复用**：使用pytest fixture来复用测试数据
- **隔离测试**：确保测试之间相互独立

### 性能测试规范

#### 1. 基准测试
- **关键路径**：对所有核心算法进行基准测试
- **内存使用**：监控内存使用情况，检测内存泄漏
- **并发性能**：测试多线程/协程场景下的性能
- **响应时间**：确保关键操作的响应时间在可接受范围内

#### 2. 性能测试工具
```python
import time
import pytest
from memory_profiler import profile

class TestPerformance:
    def test_order_book_processing_performance(self):
        """测试订单簿处理性能"""
        start_time = time.time()
        # 执行测试逻辑
        processing_time = time.time() - start_time
        assert processing_time < 0.1  # 100ms以内完成

    @profile
    def test_memory_usage(self):
        """测试内存使用情况"""
        # 内存密集型操作测试
        pass
```

#### 3. 性能监控
- **持续监控**：建立性能监控仪表板
- **回归测试**：确保代码更改不会导致性能退化
- **压力测试**：测试系统在高负载下的表现

### 集成测试要求

#### 1. 测试环境
- **隔离环境**：使用独立的测试环境
- **数据准备**：准备标准化的测试数据集
- **环境清理**：测试后清理所有临时数据和状态

#### 2. 集成场景
- **API集成**：测试与交易所API的集成
- **数据库集成**：测试数据存储和检索功能
- **消息队列**：测试异步消息处理
- **端到端流程**：测试完整的交易流程

#### 3. 集成测试示例
```python
class TestIntegration:
    def test_order_flow_to_signal_pipeline(self):
        """测试从订单流到交易信号的完整流程"""
        # 准备测试数据
        order_flow_data = create_test_order_flow()

        # 执行完整流程
        analyzer = OrderFlowAnalyzer()
        signals = analyzer.analyze(order_flow_data)

        # 验证结果
        assert len(signals) > 0
        assert all(signal.is_valid() for signal in signals)
```

### 代码审查流程

#### 1. 强制代码审查
- **python-code-reviewer Agent**：所有代码提交前必须通过python-code-reviewer agent审查
- **质量门控**：代码质量评分必须达到90分以上才能合并
- **自动触发**：在CI/CD流程中自动触发代码审查

#### 2. 审查标准
- **代码质量**：90分以上（包含可读性、维护性、性能等）
- **测试覆盖率**：单元测试覆盖率达到90%以上
- **类型安全**：通过mypy类型检查
- **安全检查**：通过安全漏洞扫描
- **文档完整性**：所有公共API都有完整文档

#### 3. 审查流程
```bash
# 1. 开发者提交代码
git add .
git commit -m "feat: add new feature"

# 2. 自动触发代码审查
# CI/CD pipeline runs:
# - ruff check .
# - mypy .
# - pytest --cov=.
# - python-code-reviewer agent

# 3. 审查不通过的处理
# 如果评分 < 90分，必须修复问题后重新提交
```

#### 4. 代码审查Checklist
- [ ] 代码符合项目编码规范
- [ ] 所有函数都有类型注解
- [ ] 异常处理完整且合理
- [ ] 单元测试覆盖率达到90%
- [ ] 性能测试通过基准要求
- [ ] 集成测试验证关键流程
- [ ] 安全扫描无高危漏洞
- [ ] 文档更新完整
- [ ] python-code-reviewer评分 ≥ 90分

## 开发命令

### 环境设置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 代码质量检查
```bash
# 代码格式化
ruff format .

# 代码检查
ruff check .

# 类型检查
mypy .

# 运行所有检查
ruff check . && mypy .
```

### 测试命令
```bash
# 运行所有测试
pytest

# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 性能测试
pytest tests/performance/

# 测试覆盖率报告
pytest --cov=. --cov-report=html

# 详细测试输出
pytest -v
```

### 代码审查
```bash
# 触发python-code-reviewer agent
# 通过CI/CD自动执行，或手动运行：
python -m claude_code.cli --review

# 检查审查结果
cat .claude/review_report.json
```

### 运行应用程序
```bash
# 运行主要交易机器人
python main.py

# 使用特定配置运行
python main.py --config config/production.yaml

# 模拟模式运行
python main.py --dry-run --backtest-days 30

# 运行特定分析组件
python -m core.order_book_analyzer --symbol BTCFDUSD
```

## 关键实现细节

### 数据处理管道
1. **实时收集**：WebSocket流用于订单簿更新和交易数据
2. **事件处理**：订单流事件被标记价格水平和市场影响
3. **水平整合**：多个数据源被聚合到价格水平桶中
4. **信号生成**：仅当静态和动态分析一致时才生成交易信号

### 做市逻辑
- **抢先策略**：将订单放置在识别到的墙前方0.01%以获得更好执行
- **吸收检测**：监控订单流以确认墙正在吸收压力
- **动态调整**：基于实时验证持续调整订单摆放

## 配置管理

系统使用`config/`目录中的YAML配置文件：
- `development.yaml`：本地开发设置
- `production.yaml`：生产交易参数
- `backtest.yaml`：历史测试配置

- 在执行pyhon命令的时候使用 venv/bin/activate && python xxx
