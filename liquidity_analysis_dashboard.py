#!/usr/bin/env python3
"""
流动性分析仪表板
提供可视化的流动性密度分析和交易监控界面
"""

import json
import redis
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiquidityAnalysisDashboard:
    """流动性分析仪表板"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """设置仪表板布局"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("BTC-FDUSD 流动性密度分析仪表板",
                           className="text-center text-primary mb-4"),
                    html.Hr()
                ])
            ]),

            # 主要指标卡片
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("当前价格", className="card-title"),
                            html.H2(id="current-price", className="text-info"),
                            html.P(id="price-change", className="card-text")
                        ])
                    ], color="dark", inverse=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("主要流动性区域", className="card-title"),
                            html.H3(id="liquidity-zone", className="text-success"),
                            html.P(id="zone-volume", className="card-text")
                        ])
                    ], color="dark", inverse=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("AI置信度", className="card-title"),
                            html.H3(id="ai-confidence", className="text-warning"),
                            html.P(id="signal-status", className="card-text")
                        ])
                    ], color="dark", inverse=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("持仓盈亏", className="card-title"),
                            html.H3(id="position-pnl", className="text-danger"),
                            html.P(id="position-info", className="card-text")
                        ])
                    ], color="dark", inverse=True)
                ], width=3)
            ], className="mb-4"),

            # 图表区域
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="volume-profile-chart")
                ], width=8),
                dbc.Col([
                    dcc.Graph(id="depth-chart")
                ], width=4)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="price-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="signal-history")
                ], width=6)
            ], className="mb-4"),

            # 交易信号和控制面板
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("实时交易信号", className="card-title"),
                            html.Div(id="latest-signals"),
                            html.Hr(),
                            html.H4("交易控制", className="card-title mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("启动交易", id="start-trading",
                                             color="success", className="me-2"),
                                    dbc.Button("停止交易", id="stop-trading",
                                             color="danger", className="me-2"),
                                    dbc.Button("强制平仓", id="force-close",
                                             color="warning")
                                ])
                            ])
                        ])
                    ], color="dark", inverse=True)
                ], width=12)
            ]),

            # 自动刷新组件
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # 10秒刷新一次
                n_intervals=0
            ),

            # 存储组件
            dcc.Store(id='market-data-store'),
            dcc.Store(id='trading-status-store')

        ], fluid=True)

    def setup_callbacks(self):
        """设置回调函数"""

        @self.app.callback(
            [Output('current-price', 'children'),
             Output('price-change', 'children'),
             Output('liquidity-zone', 'children'),
             Output('zone-volume', 'children'),
             Output('ai-confidence', 'children'),
             Output('signal-status', 'children'),
             Output('position-pnl', 'children'),
             Output('position-info', 'children'),
             Output('market-data-store', 'data'),
             Output('trading-status-store', 'data')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """更新仪表板数据"""
            try:
                # 获取市场数据
                market_data = self._get_market_data()
                trading_status = self._get_trading_status()

                if not market_data:
                    return ("无数据", "", "无数据", "", "无数据", "", "无数据", "", {}, {})

                current_price = market_data.get('current_price', 0)
                price_change_1h = market_data.get('price_change_1h', 0)
                price_change_24h = market_data.get('price_change_24h', 0)

                # 获取流动性分析
                liquidity_zone = self._get_liquidity_analysis(market_data)

                # 获取AI分析结果
                ai_analysis = self._get_ai_analysis()

                # 格式化显示
                price_display = f"${current_price:,.2f}" if current_price > 0 else "无数据"
                price_change_display = f"1h: {price_change_1h:+.2%} | 24h: {price_change_24h:+.2%}"

                if liquidity_zone:
                    zone_display = f"${liquidity_zone.get('price_center', 0):,.2f}"
                    volume_display = f"成交量: {liquidity_zone.get('total_volume', 0):.3f} BTC"
                else:
                    zone_display = "识别中..."
                    volume_display = ""

                confidence_display = f"{ai_analysis.get('confidence', 0):.1%}" if ai_analysis else "无数据"
                signal_status = ai_analysis.get('signal', '等待中') if ai_analysis else "等待中"

                pnl_display = f"${trading_status.get('realized_pnl', 0):,.2f}"
                position_info = f"持仓: {trading_status.get('current_position', 0):.6f} BTC"

                return (price_display, price_change_display, zone_display, volume_display,
                       confidence_display, signal_status, pnl_display, position_info,
                       market_data, trading_status)

            except Exception as e:
                logger.error(f"更新仪表板失败: {e}")
                return ("错误", "", "错误", "", "错误", "", "错误", "", {}, {})

        @self.app.callback(
            Output('volume-profile-chart', 'figure'),
            [Input('market-data-store', 'data')]
        )
        def update_volume_profile(market_data):
            """更新成交量分布图"""
            if not market_data:
                return go.Figure()

            try:
                trades_data = market_data.get('trades_data', [])
                if not trades_data:
                    return go.Figure()

                # 构建成交量分布
                volume_data = []
                price_levels = []

                for minute_data in trades_data[-100:]:  # 最近100分钟
                    for price_str, level_data in minute_data.get('price_levels', {}).items():
                        price = float(price_str)
                        volume = level_data.get('total_volume', 0)
                        volume_data.append(volume)
                        price_levels.append(price)

                if not volume_data:
                    return go.Figure()

                # 创建成交量分布直方图
                fig = go.Figure(data=[
                    go.Bar(
                        x=price_levels,
                        y=volume_data,
                        name='成交量',
                        marker_color='rgba(55, 128, 191, 0.7)',
                        hovertemplate='价格: $%{x:,.2f}<br>成交量: %{y:.3f} BTC<extra></extra>'
                    )
                ])

                fig.update_layout(
                    title="成交量分布图",
                    xaxis_title="价格 (USDT)",
                    yaxis_title="成交量 (BTC)",
                    template="plotly_dark",
                    height=400
                )

                return fig

            except Exception as e:
                logger.error(f"更新成交量分布图失败: {e}")
                return go.Figure()

        @self.app.callback(
            Output('depth-chart', 'figure'),
            [Input('market-data-store', 'data')]
        )
        def update_depth_chart(market_data):
            """更新深度图表"""
            if not market_data:
                return go.Figure()

            try:
                depth_data = market_data.get('depth_data', {})
                if not depth_data:
                    return go.Figure()

                asks = depth_data.get('asks', [])
                bids = depth_data.get('bids', [])

                if not asks or not bids:
                    return go.Figure()

                # 处理数据
                ask_prices, ask_volumes = zip(*asks[:20])  # 前20档卖单
                bid_prices, bid_volumes = zip(*bids[:20])  # 前20档买单

                # 计算累积成交量
                ask_volumes_cumsum = np.cumsum(ask_volumes)[::-1]
                bid_volumes_cumsum = np.cumsum(bid_volumes)

                fig = go.Figure()

                # 添加卖单深度
                fig.add_trace(go.Scatter(
                    x=ask_prices,
                    y=ask_volumes_cumsum,
                    mode='lines',
                    name='卖单深度',
                    line=dict(color='red'),
                    fill='tonexty'
                ))

                # 添加买单深度
                fig.add_trace(go.Scatter(
                    x=bid_prices,
                    y=bid_volumes_cumsum,
                    mode='lines',
                    name='买单深度',
                    line=dict(color='green'),
                    fill='tonexty'
                ))

                fig.update_layout(
                    title="订单簿深度",
                    xaxis_title="价格 (USDT)",
                    yaxis_title="累积成交量 (BTC)",
                    template="plotly_dark",
                    height=400,
                    showlegend=True
                )

                return fig

            except Exception as e:
                logger.error(f"更新深度图表失败: {e}")
                return go.Figure()

        @self.app.callback(
            Output('price-chart', 'figure'),
            [Input('market-data-store', 'data')]
        )
        def update_price_chart(market_data):
            """更新价格走势图"""
            if not market_data:
                return go.Figure()

            try:
                trades_data = market_data.get('trades_data', [])
                if not trades_data:
                    return go.Figure()

                # 提取价格时间序列
                timestamps = []
                prices = []
                volumes = []

                for minute_data in trades_data[-200:]:  # 最近200分钟
                    timestamp = minute_data.get('timestamp', '')
                    price_levels = minute_data.get('price_levels', {})

                    if price_levels:
                        # 计算成交量加权平均价格
                        total_volume = sum(item['total_volume'] for item in price_levels.values())
                        if total_volume > 0:
                            weighted_price = sum(
                                float(price) * item['total_volume']
                                for price, item in price_levels.items()
                            ) / total_volume

                            timestamps.append(timestamp)
                            prices.append(weighted_price)
                            volumes.append(total_volume)

                if not prices:
                    return go.Figure()

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=('价格走势', '成交量'),
                    row_heights=[0.7, 0.3]
                )

                # 价格走势
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=prices,
                        mode='lines',
                        name='价格',
                        line=dict(color='cyan')
                    ),
                    row=1, col=1
                )

                # 成交量
                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=volumes,
                        name='成交量',
                        marker_color='rgba(255, 165, 0, 0.7)'
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    title="价格走势与成交量",
                    template="plotly_dark",
                    height=500,
                    showlegend=False
                )

                return fig

            except Exception as e:
                logger.error(f"更新价格图表失败: {e}")
                return go.Figure()

        @self.app.callback(
            Output('signal-history', 'figure'),
            [Input('trading-status-store', 'data')]
        )
        def update_signal_history(trading_status):
            """更新信号历史图"""
            if not trading_status:
                return go.Figure()

            try:
                # 这里应该从Redis获取交易历史
                # 简化版实现
                fig = go.Figure()

                # 示例数据
                fig.add_trace(go.Scatter(
                    x=[datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)],
                    y=np.random.randn(24).cumsum(),
                    mode='lines+markers',
                    name='权益曲线',
                    line=dict(color='lime')
                ))

                fig.update_layout(
                    title="权益曲线",
                    xaxis_title="时间",
                    yaxis_title="权益 (USDT)",
                    template="plotly_dark",
                    height=400
                )

                return fig

            except Exception as e:
                logger.error(f"更新信号历史图失败: {e}")
                return go.Figure()

        @self.app.callback(
            Output('latest-signals', 'children'),
            [Input('trading-status-store', 'data')]
        )
        def update_latest_signals(trading_status):
            """更新最新交易信号"""
            if not trading_status:
                return dbc.Alert("等待交易信号...", color="info")

            signals = []

            # 显示当前信号
            current_signal = trading_status.get('latest_signal')
            if current_signal:
                signal_color = {
                    'BUY': 'success',
                    'SELL': 'danger',
                    'HOLD': 'secondary'
                }.get(current_signal.get('action', 'HOLD'), 'secondary')

                signals.append(
                    dbc.Alert([
                        html.H6(f"信号: {current_signal.get('action', 'UNKNOWN')}"),
                        html.P(f"价格: ${current_signal.get('price', 0):.2f}"),
                        html.P(f"置信度: {current_signal.get('confidence', 0):.1%}"),
                        html.P(f"原因: {current_signal.get('reason', '')}")
                    ], color=signal_color)
                )

            return signals

    def _get_market_data(self) -> Dict:
        """获取市场数据"""
        try:
            # 从Redis获取数据
            trades_data = self._get_recent_trades()
            depth_data = self._get_depth_snapshot()

            if not trades_data:
                return {}

            current_price = self._get_current_price(trades_data[-1])

            return {
                'trades_data': trades_data,
                'depth_data': depth_data,
                'current_price': current_price,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return {}

    def _get_recent_trades(self) -> List[Dict]:
        """获取最近交易数据"""
        try:
            trades = []
            window_size = min(200, self.redis.llen('trades_window'))

            for i in range(window_size):
                data_str = self.redis.lindex('trades_window', -i-1)
                if data_str:
                    data = json.loads(data_str)
                    trades.append(data)

            return trades[::-1]

        except Exception as e:
            logger.error(f"获取交易数据失败: {e}")
            return []

    def _get_depth_snapshot(self) -> Optional[Dict]:
        """获取深度快照"""
        try:
            depth_str = self.redis.get('depth_snapshot_5000')
            if depth_str:
                return json.loads(depth_str)
            return None

        except Exception as e:
            logger.error(f"获取深度快照失败: {e}")
            return None

    def _get_current_price(self, minute_data: Dict) -> float:
        """获取当前价格"""
        price_levels = minute_data.get('price_levels', {})
        if not price_levels:
            return 0.0

        max_volume_price = max(
            price_levels.keys(),
            key=lambda p: price_levels[p]['total_volume']
        )
        return float(max_volume_price)

    def _get_trading_status(self) -> Dict:
        """获取交易状态"""
        try:
            # 从Redis获取交易状态
            status_str = self.redis.lindex('trading_status_log', 0)
            if status_str:
                return json.loads(status_str)
            return {}

        except Exception as e:
            logger.error(f"获取交易状态失败: {e}")
            return {}

    def _get_liquidity_analysis(self, market_data: Dict) -> Optional[Dict]:
        """获取流动性分析"""
        try:
            # 这里应该调用流动性分析器
            # 简化版实现
            return {
                'price_center': market_data.get('current_price', 0),
                'total_volume': 1.0,
                'buy_ratio': 0.6,
                'sell_ratio': 0.4
            }

        except Exception as e:
            logger.error(f"获取流动性分析失败: {e}")
            return None

    def _get_ai_analysis(self) -> Optional[Dict]:
        """获取AI分析"""
        try:
            # 这里应该从AI系统获取分析结果
            # 简化版实现
            return {
                'confidence': 0.75,
                'signal': 'BUY',
                'entry_price': 114500.0,
                'stop_loss': 114000.0,
                'take_profit': 115000.0
            }

        except Exception as e:
            logger.error(f"获取AI分析失败: {e}")
            return None

    def run(self, debug=False, port=8050):
        """运行仪表板"""
        logger.info(f"启动流动性分析仪表板 - http://localhost:{port}")
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')

def main():
    """主函数"""
    # 连接Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # 创建并运行仪表板
    dashboard = LiquidityAnalysisDashboard(redis_client)
    dashboard.run(debug=True)

if __name__ == "__main__":
    main()