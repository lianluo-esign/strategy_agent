"""配置常量模块。

这个模块定义了系统中使用的各种配置常量，避免魔法数字分散在代码中。
"""

# API配置常量
DEFAULT_API_TIMEOUT = 90  # 默认API超时时间（秒）
DEFAULT_MAX_RETRIES = 3   # 默认最大重试次数
DEFAULT_MAX_TOKENS = 6000  # 默认最大令牌数
DEFAULT_TEMPERATURE = 0.1  # 默认温度设置

# 数据处理常量
DEFAULT_BID_LEVELS = 15   # 默认处理的买盘档位数
DEFAULT_ASK_LEVELS = 15   # 默认处理的卖盘档位数
DEFAULT_VP_LEVELS = 10    # 默认处理的Volume Price档位数

# 价格范围常量
PRICE_LEVEL_BILLION = 1e9      # 十亿价位阈值
PRICE_LEVEL_HUNDRED_MILLION = 1e8  # 一亿价位阈值
PRICE_LEVEL_TEN_MILLION = 1e7     # 千万价位阈值
PRICE_LEVEL_MILLION = 1e6         # 百万价位阈值
PRICE_LEVEL_HUNDRED_THOUSAND = 1e5    # 十万价位阈值
PRICE_LEVEL_TEN_THOUSAND = 1e4        # 万价位阈值
PRICE_LEVEL_THOUSAND = 1e3           # 千价位阈值
PRICE_LEVEL_HUNDRED = 1e2            # 百价位阈值

# 验证常量
MAX_SYMBOL_LENGTH = 20     # 交易符号最大长度
MIN_PRICE_VALUE = 0        # 最小价格值
MAX_VOLUME_VALUE = 1e15    # 最大成交量值

# 重试配置
RETRY_MULTIPLIER = 1       # 重试间隔倍数
RETRY_MIN_DELAY = 4        # 最小重试延迟（秒）
RETRY_MAX_DELAY = 10       # 最大重试延迟（秒）

# 日志配置
MAX_LOG_LINE_LENGTH = 120  # 最大日志行长度
MAX_CONTENT_PREVIEW_LINES = 10  # 最大内容预览行数

# 分析模式常量
ANALYSIS_MODE_UNIFIED = "unified"
ANALYSIS_MODE_TRADITIONAL = "traditional"
ANALYSIS_MODE_DISABLED = "disabled"

# 错误消息常量
ERROR_INVALID_API_KEY = "DeepSeek API密钥是必需的"
ERROR_INVALID_SYMBOL = "Symbol must be a non-empty string"
ERROR_INVALID_BIDS = "Aggregated bids must be a dictionary"
ERROR_INVALID_ASKS = "Aggregated asks must be a dictionary"
ERROR_INVALID_VP_RESULT = "VP result must be a dictionary"
ERROR_MISSING_VP_DATA = "VP result must contain 'vp_data' field"
ERROR_INVALID_DECIMAL_TYPE = "must be Decimal instances"
ERROR_NEGATIVE_VOLUME = "volumes cannot be negative"
ERROR_INVALID_RESPONSE_STRUCTURE = "响应结构解析失败"
ERROR_PARSING_FAILED = "解析失败"

# HTTP状态码常量
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

# JSON解析常量
JSON_START_MARKER = "{"
JSON_END_MARKER = "}"

# 格式化常量
CURRENCY_SYMBOL = "$"
THOUSANDS_SEPARATOR = ","
DECIMAL_SEPARATOR = "."