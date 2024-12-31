# === SECTION 1: IMPORTS AND SETUP ===
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import json
import os
import sys
import requests
from datetime import datetime, timedelta, time as dt_time
from py_vollib_vectorized import price_dataframe, get_all_greeks, vectorized_implied_volatility
import numpy as np
import pytz
from scipy.stats import norm
import traceback
import math
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings("ignore", message="Found Below Intrinsic contracts at index")

# === SECTION 2: CONSTANTS AND GLOBAL VARIABLES ===
state_colors = {
    0: "#FFFFFF",  # White for undefined state
    1: "#32CD32",  # Lime Green
    2: "#FFFF00",  # Yellow
    3: "#FFA500",  # Orange
    4: "#FF0000"   # Bright Red
}

CUSTOM_SYMBOLS_FILE = 'custom_symbols.json'

# Global variables for historical price tracking
historical_prices = []
last_price_update = datetime.now()
data_cache = {}

# === SECTION 3: APP INITIALIZATION ===
app = dash.Dash(
    __name__, 
    title="MoonShotFlows",
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

# === SECTION 4: APP INDEX STRING ===
app.index_string = '''
<!DOCTYPE html>
<html style="margin:0; padding:0; width:100vw; overflow-x:hidden;">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Essential Dropdown Styles with improved visibility */
            .Select-control {
                background-color: #333333 !important;
                border-color: #666 !important;
                height: 40px !important;
                color: white !important;
            }

            .Select-menu-outer {
                background-color: #333333 !important;
                border: 1px solid #666 !important;
                color: white !important;
                position: absolute !important;
                z-index: 9999 !important;
            }

            .Select-menu {
                max-height: 300px !important;
                overflow-y: auto !important;
            }

            .Select-option {
                background-color: #333333 !important;
                color: white !important;
                padding: 8px 12px !important;
            }

            .Select-option:hover {
                background-color: #444444 !important;
                color: white !important;
            }

            .Select-value {
                line-height: 40px !important;
            }

            .Select-value-label {
                color: white !important;
                line-height: 40px !important;
            }

            .Select-placeholder {
                color: #CCCCCC !important;
                line-height: 40px !important;
            }

            .Select-input > input {
                color: white !important;
                padding: 8px 0 !important;
            }

            .Select.is-focused > .Select-control {
                background-color: #333333 !important;
                border-color: #888 !important;
            }

            .Select.is-focused:not(.is-open) > .Select-control {
                border-color: #888 !important;
                box-shadow: none !important;
            }

            .Select-arrow {
                border-color: white transparent transparent !important;
            }

            .Select-arrow-zone:hover > .Select-arrow {
                border-top-color: #CCCCCC !important;
            }

            .Select.is-focused > .Select-control .Select-placeholder {
                color: white !important;
            }

            .Select-clear-zone {
                color: #CCCCCC !important;
            }

            .Select-clear-zone:hover {
                color: white !important;
            }

            .Select.has-value.is-focused > .Select-control .Select-value .Select-value-label,
            .Select.has-value > .Select-control .Select-value .Select-value-label {
                color: white !important;
            }

            /* Scrollbar Styles */
            ::-webkit-scrollbar {
                width: 8px;
                background-color: transparent;
            }
            
            ::-webkit-scrollbar-track {
                background-color: transparent;
            }
            
            ::-webkit-scrollbar-thumb {
                background-color: rgba(0,0,0,0.5);
                border-radius: 4px;
                border: 1px solid rgba(50,50,50,0.5);
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background-color: rgba(30,30,30,0.8);
            }
            
            /* Firefox scrollbar styling */
            * {
                scrollbar-width: thin;
                scrollbar-color: rgba(0,0,0,0.5) transparent;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_dropdown(id, options, value, width='100px', margin_right='10px'):
    """Helper function to create consistently styled dropdowns"""
    return dcc.Dropdown(
        id=id,
        options=options,
        value=value,
        style={
            **DROPDOWN_STYLE,
            'width': width,
            'marginRight': margin_right
        }
    )

def create_dashboard_layout():
    return html.Div([
        dcc.Interval(
            id='interval-component',
            interval=5000,
            n_intervals=0
        ),
        
        # Header (unchanged)
        html.Div([
            html.Div([
                html.Div([
                    dcc.Input(
                        id='symbol-input',
                        type='text',
                        placeholder='Symbol...',
                        style={
                            'width': '65px',
                            'backgroundColor': '#333333',
                            'color': 'white',
                            'border': '1px solid #666',
                            'borderRadius': '4px',
                            'padding': '5px',
                            'marginRight': '5px',
                            'height': '40px'
                        }
                    ),
                    html.Button(
                        '+',
                        id='add-symbol-button',
                        style={
                            'backgroundColor': '#444444',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',
                            'padding': '6px 8px',
                            'marginRight': '1px',
                            'cursor': 'pointer',
                            'width': '35px',
                            'height': '40px'
                        }
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '5px'}),
                
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=[
                        {'label': symbol, 'value': symbol} 
                        for symbol in ['SPY', 'QQQ', 'IWM', 'AAPL', 'AMZN', 
                                     'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
                    ],
                    value='SPY',
                    style={
                        **DROPDOWN_STYLE,
                        'width': '100px',
                        'marginRight': '5px'
                    }
                ),
                dcc.Dropdown(
                    id='date-dropdown',
                    style={
                        **DROPDOWN_STYLE,
                        'width': '145px',
                        'marginLeft': '0px'
                    }
                ),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginLeft': '20px'
            }),
            
            html.Div(id='header-price', children=[], style={
                'color': 'white',
                'fontSize': '28px',
                'fontWeight': 'bold',
                'flex': '1',
                'textAlign': 'center',
                'marginRight': '345px',
                'position': 'relative'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '5px',
            'backgroundColor': '#1a1a1a',
            'position': 'fixed',
            'top': '40px',
            'left': 0,
            'right': 0,
            'width': '100%',
            'zIndex': 997,
            'height': '65px',
            'margin': 0
        }),

        # Main Content Container
        html.Div([
            # Left Panel - Price Charts
            html.Div([
                html.Div(id='price-charts', children=[], style={
                    'backgroundColor': 'black',
                    'minHeight': '1500px',
                    'marginRight': '20px',
                    'border': 'None',
                    'borderRadius': '5px',
                    'position': 'relative'
                })
            ], style={
                'width': '42%',
                'backgroundColor': 'black',
                'padding': '20px',
                'height': 'calc(100vh - 105px)',
                'overflowY': 'auto',
                'msOverflowStyle': 'none',
                'scrollbarWidth': 'none',
                '::-webkit-scrollbar': {'display': 'none'},
                'marginTop': '89px'
            }),

            # Center Panel - Metrics, Gauges, and States
            html.Div([
                # Metrics Container
                html.Div(id='output-container', children=[], style={
                    'padding': '15px',
                    'backgroundColor': 'black',
                    'marginTop': '0px',
                    'marginLeft': '0px',
                    'minHeight': '200px',
                    'position': 'relative'
                }),
                
                # Gauges Container
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='gex-gauge-chart',
                            config={'displayModeBar': False},
                            style={
                                'height': '180px',
                                'marginLeft': '0px',
                                'marginBottom': '75px',
                                'position': 'relative'
                            }
                        )
                    ], style={'minHeight': '255px'}),
                    html.Div([
                        dcc.Graph(
                            id='vex-gauge-chart',
                            config={'displayModeBar': False},
                            style={
                                'height': '180px',
                                'marginLeft': '0px',
                                'marginBottom': '75px',
                                'position': 'relative'
                            }
                        )
                    ], style={'minHeight': '255px'}),
                    html.Div([
                        dcc.Graph(
                            id='dex-gauge-chart',
                            config={'displayModeBar': False},
                            style={
                                'height': '180px',
                                'marginLeft': '0px',
                                'marginBottom': '20px',
                                'position': 'relative'
                            }
                        )
                    ], style={'minHeight': '200px'})
                ], style={
                    'backgroundColor': 'black',
                    'padding': '0',
                    'marginTop': '-40px',
                    'minHeight': '800px'
                }),
                
                # States Section
                html.Div([
                    # State 1
                    html.Details([
                        html.Summary('State 1: +GEX/+VEX (Balanced)', 
                                   id='state1-title', 
                                   style={'color': '#32CD32'}),
                        html.Div([
                            html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Upward moves trigger increasing selling pressure', style={'color': 'white'}),
                                html.Li('Downward moves trigger increasing buying pressure', style={'color': 'white'}),
                                html.Li('Price tends to stabilize between significant GEX levels', style={'color': 'white'})
                            ]),
                            html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Primary Setup: Identify largest green GEX bars as key levels', style={'color': 'white'}),
                                html.Li('Entry Points: Look for trades between major GEX strikes', style={'color': 'white'}),
                                html.Li('Price Below Flip Point: Expect bounce to first positive strike', style={'color': 'white'}),
                                html.Li('Best Opportunity: Range-bound trading between strong GEX levels', style={'color': 'white'})
                            ])
                        ], style={'padding': '10px'})
                    ], style={'marginBottom': '10px'}),

                    # State 2
                    html.Details([
                        html.Summary('State 2: +GEX/-VEX (Trending)', 
                                   id='state2-title', 
                                   style={'color': '#FFFF00'}),
                        html.Div([
                            html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Upward moves trigger increased buying pressure', style={'color': 'white'}),
                                html.Li('Downward moves trigger increased selling pressure', style={'color': 'white'}),
                                html.Li('Market tends toward trending movement', style={'color': 'white'})
                            ]),
                            html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Primary Setup: Monitor VEX chart for key levels', style={'color': 'white'}),
                                html.Li('Entry Points: Trade in direction of the break', style={'color': 'white'}),
                                html.Li('Risk Management: Use VEX levels as targets', style={'color': 'white'}),
                                html.Li('Best Opportunity: Momentum trades following breakouts', style={'color': 'white'})
                            ])
                        ], style={'padding': '10px'})
                    ], style={'marginBottom': '10px'}),

                    # State 3
                    html.Details([
                        html.Summary('State 3: -GEX/+VEX (Reversing)', 
                                   id='state3-title', 
                                   style={'color': '#FFA500'}),
                        html.Div([
                            html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Upward moves attract more sellers', style={'color': 'white'}),
                                html.Li('Price gravitates toward largest GEX strike', style={'color': 'white'}),
                                html.Li('Significant VEX levels influence price action', style={'color': 'white'})
                            ]),
                            html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Primary Setup: Short at closest positive GEX strike', style={'color': 'white'}),
                                html.Li('Entry Points: Look for price rejection at large VEX levels', style={'color': 'white'}),
                                html.Li('Risk Management: Use major GEX strikes as targets', style={'color': 'white'}),
                                html.Li('Best Opportunity: Short positions near significant resistance levels', style={'color': 'white'})
                            ])
                        ], style={'padding': '10px'})
                    ], style={'marginBottom': '10px'}),

                    # State 4
                    html.Details([
                        html.Summary('State 4: -GEX/-VEX (Volatile)', 
                                   id='state4-title', 
                                   style={'color': '#FF0000'}),
                        html.Div([
                            html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Downward moves trigger accelerated selling pressure', style={'color': 'white'}),
                                html.Li('Upward bounces are typically weak', style={'color': 'white'}),
                                html.Li('Price tends to move rapidly until reaching major GEX level', style={'color': 'white'})
                            ]),
                            html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Primary Setup: Look for confluence of large GEX and VEX levels', style={'color': 'white'}),
                                html.Li('Entry Points: Watch for price reaction at largest GEX strike levels', style={'color': 'white'}),
                                html.Li('Risk Management: Use VEX levels for potential support/resistance points', style={'color': 'white'}),
                                html.Li('Best Opportunity: Trade reversals when price reaches significant GEX levels', style={'color': 'white'})
                            ])
                        ], style={'padding': '10px'})
                    ], style={'marginBottom': '10px'}),

                    # Vega Summary
                    html.Details([
                        html.Summary('Vega Summary', style={'color': 'white'}),
                        html.Div([
                            html.P('Market State Indicators:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Positive Total Vega (Long): Shows market positioning for volatility expansion', style={'color': 'white'}),
                                html.Li('Negative Total Vega (Short): Shows market positioning for volatility contraction', style={'color': 'white'}),
                                html.Li('VEX Chart: Maps key volatility exposure levels that can act as price inflection points', style={'color': 'white'})
                            ]),
                            html.P('Trading Applications:', style={'color': 'white', 'fontWeight': 'bold'}),
                            html.Ul([
                                html.Li('Large VEX strikes often act as significant price inflection points', style={'color': 'white'}),
                                html.Li('High positive VEX levels: Expect increased volatility and possible resistance', style={'color': 'white'}),
                                html.Li('High negative VEX levels: Expect decreased volatility and possible support', style={'color': 'white'}),
                                html.Li('Most reliable signals occur when major GEX and VEX levels align', style={'color': 'white'})
                            ])
                        ], style={'padding': '10px'})
                    ])
                ], style={
                    'backgroundColor': 'black',
                    'padding': '20px',
                    'marginTop': '20px',
                    'border': '1px solid #333',
                    'borderRadius': '5px'
                })
            ], style={
                'width': '20%',
                'backgroundColor': 'black',
                'padding': '20px 20px 0 0',
                'height': 'calc(100vh - 105px)',
                'overflowY': 'auto',
                'msOverflowStyle': 'none',
                'scrollbarWidth': 'none',
                '::-webkit-scrollbar': {'display': 'none'},
                'marginTop': '53px'
            }),

            # Right Panel section with completed style definitions
            # Right Panel - Histograms only
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='gex-histogram',
                            config={'displayModeBar': False},
                            style={
                                'height': '300px',
                                'marginBottom': '20px',
                                'marginTop': '20px',
                                'marginRight': '30px',
                                'border': '1px solid #333',
                                'borderRadius': '5px',
                                'position': 'relative'
                            }
                        )
                    ], style={'minHeight': '320px'}),
                    html.Div([
                        dcc.Graph(
                            id='vex-histogram',
                            config={'displayModeBar': False},
                            style={
                                'height': '300px',
                                'marginBottom': '20px',
                                'marginRight': '30px',
                                'border': '1px solid #333',
                                'borderRadius': '5px',
                                'position': 'relative'
                            }
                        )
                    ], style={'minHeight': '320px'}),
                    html.Div([
                        dcc.Graph(
                            id='dex-histogram',
                            config={'displayModeBar': False},
                            style={
                                'height': '300px',
                                'marginBottom': '10px',
                                'marginRight': '30px',
                                'border': '1px solid #333',
                                'borderRadius': '5px',
                                'position': 'relative'
                            }
                        )
                    ], style={'minHeight': '330px'})
                ], style={
                    'backgroundColor': 'black',
                    'width': '100%',
                    'marginBottom': '40px',
                    'marginRight': '30px',
                    'minHeight': '1000px'
                })
            ], style={
                'width': '35%',
                'backgroundColor': 'black',
                'padding': '20px 0 20px 20px',
                'height': 'calc(100vh - 105px)',
                'overflowY': 'auto',
                'msOverflowStyle': 'none',
                'scrollbarWidth': 'none',
                '::-webkit-scrollbar': {'display': 'none'},
                'marginTop': '65px'
            })
        ], style={
            'display': 'flex',
            'backgroundColor': 'black',
            'minHeight': '100vh',
            'margin': 0,
            'padding': 0,
            'width': '100%',
            'overflowX': 'hidden',
            'position': 'relative'
        })
    ], style={
        'backgroundColor': 'black',
        'minHeight': '100vh',
        'width': '100%',
        'margin': 0,
        'padding': 0,
        'border': 'none',
        'overflowY': 'hidden',
        'overflowX': 'hidden',
        'position': 'relative'
    })

def create_scrollable_panel_style(width, include_margin=True):
    """Create consistent panel styling with transparent scrollbars"""
    style = {
        'width': width,
        'padding': '20px',
        'overflowY': 'auto',
        'height': 'calc(100vh - 105px)',
        'paddingBottom': '40px',
        'backgroundColor': 'black',
        'scrollbarWidth': 'thin',
        'scrollbarColor': 'rgba(0,0,0,0.5) transparent',
        'msOverflowStyle': 'none',  # IE and Edge
        'scrollbarWidth': 'none'    # Firefox
    }
    
    if include_margin:
        style['marginTop'] = '105px'
        
    return style

DROPDOWN_STYLE = {
    'backgroundColor': '#333333',
    'color': 'white',
    'height': '40px',
    'border': '1px solid #666',
    'borderRadius': '4px',
    'fontSize': '14px',
    'zIndex': 500
}

def create_analysis_layout():
    # Add this at the beginning of the function
    def get_all_symbols():
        default_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
        custom_symbols = load_custom_symbols()
        all_symbols = list(set(default_symbols + custom_symbols))  # Remove duplicates
        return sorted(all_symbols)  # Sort alphabetically

    all_symbols = get_all_symbols()
    
    return html.Div([
        dcc.Interval(
            id='analysis-interval-component',
            interval=5000,
            n_intervals=0
        ),
        # Header Bar
        html.Div([
            html.Div([
                # Symbol Dropdown
                dcc.Dropdown(
                    id='analysis-symbol-dropdown',
                    options=[{'label': s, 'value': s} for s in get_all_symbols()],
                    value=all_symbols[0] if all_symbols else 'SPY',
                    style={
                        'width': '120px',
                        'backgroundColor': '#333333',
                        'color': 'white',
                        'marginRight': '10px'
                    }
                ),
                # Timeframe Dropdown
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[
                        {'label': '1D', 'value': '1D'},
                        {'label': '2D', 'value': '2D'},
                        {'label': '3D', 'value': '3D'},
                        {'label': '4D', 'value': '4D'},                        
                        {'label': '1W', 'value': '1W'},
                        {'label': '1M', 'value': '1M'},
                        {'label': '3M', 'value': '3M'},
                        {'label': '6M', 'value': '6M'},
                        {'label': '1Y', 'value': '1Y'}
                    ],
                    value='1D',
                    style={
                        'width': '100px',
                        'backgroundColor': '#333333',
                        'color': 'white',
                        'marginRight': '10px'
                    }
                ),
                # Interval Dropdown
                dcc.Dropdown(
                    id='interval-dropdown',
                    options=[
                        {'label': '1 min', 'value': '1min'},
                        {'label': '5 min', 'value': '5min'},
                        {'label': '15 min', 'value': '15min'},
                        {'label': '30 min', 'value': '30min'},
                        {'label': '1 hour', 'value': '1hour'},
                        {'label': '2 hour', 'value': '2hour'},
                        {'label': '4 hour', 'value': '4hour'},
                        {'label': 'Daily', 'value': 'daily'}
                    ],
                    value='5min',
                    style={
                        'width': '100px',
                        'backgroundColor': '#333333',
                        'color': 'white',
                        'marginRight': '10px'
                    }
                ),
                # Go Button
                html.Button(
                    'Go',
                    id='analysis-go-button',
                    style={
                        'backgroundColor': '#0d71df',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '8px 15px',
                        'cursor': 'pointer',
                        'height': '38px',
                        'marginLeft': '5px'
                    }
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginLeft': '20px'
            }),
            # Header Price Display
            html.Div(id='analysis-header-price', style={
                'color': 'white',
                'fontSize': '28px',
                'fontWeight': 'bold',
                'flex': '1',
                'textAlign': 'center',
                'marginRight': '405px'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '10px 5px',
            'backgroundColor': '#1a1a1a',
            'position': 'fixed',
            'top': '40px',
            'left': 0,
            'right': 0,
            'zIndex': 1000,
            'height': '65px'
        }),

        # Main Content Area
        html.Div([
            # Left Panel - Charts
            html.Div([
                # Price Chart
                html.Div([
                    dcc.Graph(
                        id='analysis-price-chart',
                        config={'scrollZoom': True, 'displayModeBar': False}
                    )
                ], style={
                    'backgroundColor': '#1a1a1a',
                    'padding': '5px',
                    'borderRadius': '5px',
                    'marginBottom': '20px',
                    'border': '1px solid #333'
                }),
                # RSI Chart
                html.Div([
                    dcc.Graph(
                        id='rsi-chart',
                        config={'scrollZoom': True, 'displayModeBar': False}
                    )
                ], style={
                    'backgroundColor': '#1a1a1a',
                    'padding': '5px',
                    'borderRadius': '5px',
                    'marginBottom': '20px',
                    'border': '1px solid #333'
                })
            ], style={
                'width': '75%',
                'padding': '20px',
                'height': 'calc(100vh - 105px)',
                'overflowY': 'auto',
                'overflowX': 'hidden',
                'marginTop': '108px',
                'scrollbarWidth': 'thin',
                'scrollbarColor': 'rgba(0,0,0,0.5) transparent'
            }),

            # Right Panel - Technical Analysis
            html.Div([
                html.Div([
                    # Technical Indicators Section
                    html.Div([
                        html.H3('Technical Indicators', style={'color': 'white', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div('Moving Averages', style={'color': '#888', 'marginBottom': '10px'}),
                            html.Div(id='ma-signals', style={'marginBottom': '20px'})
                        ]),
                        html.Div([
                            html.Div('Oscillators', style={'color': '#888', 'marginBottom': '10px'}),
                            html.Div(id='oscillator-signals', style={'marginBottom': '20px'})
                        ]),                        
                        html.Div([
                            html.Div('Pivot Points', style={'color': '#888', 'marginBottom': '10px'}),
                            html.Div(id='pivot-points', style={'marginBottom': '20px'})
                        ])
                    ], style={
                        'backgroundColor': '#1a1a1a',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'marginBottom': '20px'
                    }),

                    # Key Levels Section
                    html.Div([
                        html.H3('Key Levels', style={'color': 'white', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div('Support', style={'color': '#888', 'marginBottom': '10px'}),
                            html.Div(id='support-levels', style={'marginBottom': '20px'})
                        ]),
                        html.Div([
                            html.Div('Resistance', style={'color': '#888', 'marginBottom': '10px'}),
                            html.Div(id='resistance-levels', style={'marginBottom': '20px'})
                        ])
                    ], style={
                        'backgroundColor': '#1a1a1a',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'marginBottom': '20px'
                    }),

                    # Options Analysis Section
                    html.Div([
                        html.H3('Options Analysis', style={'color': 'white', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div('IV Percentile', style={'color': '#888', 'marginBottom': '10px'}),
                            html.Div(id='iv-percentile', style={'marginBottom': '20px'})
                        ]),
                        html.Div([
                            html.Div('Put/Call Ratio', style={'color': '#888', 'marginBottom': '10px'}),
                            html.Div(id='put-call-ratio', style={'marginBottom': '20px'})
                        ])
                    ], style={
                        'backgroundColor': '#1a1a1a',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'marginBottom': '20px'
                    })
                ], style=create_scrollable_panel_style('100%'))
            ], style={
                'width': '23%',
                'padding': '0 20px'
            })
        ], style={
            'display': 'flex',
            'backgroundColor': 'black',
            'height': '100vh',
            'overflowY': 'hidden'
        })
    ], style={
        'backgroundColor': 'black',
        'height': '100vh',
        'width': '100vw',
        'position': 'relative',
        'overflow': 'hidden'
    })

def create_scanner_layout():
    return html.Div([
        # Header Section
        html.Div([
            html.Div([
                # Scanner Type Dropdown
                dcc.Dropdown(
                    id='scanner-type-dropdown',
                    options=[
                        {'label': 'Unusual Options Activity', 'value': 'uoa'},
                        {'label': 'High IV Rank', 'value': 'iv-rank'},
                        {'label': 'Momentum Signals', 'value': 'momentum'},
                        {'label': 'Technical Breakouts', 'value': 'breakouts'}
                    ],
                    value='uoa',
                    style={
                        'width': '200px',
                        'backgroundColor': '#333333',
                        'color': 'white',
                        'marginRight': '10px'
                    }
                ),
                # Market Filter
                dcc.Dropdown(
                    id='market-filter-dropdown',
                    options=[
                        {'label': 'All Markets', 'value': 'all'},
                        {'label': 'S&P 500', 'value': 'sp500'},
                        {'label': 'NASDAQ 100', 'value': 'ndx'},
                        {'label': 'Russell 2000', 'value': 'rut'}
                    ],
                    value='all',
                    style={
                        'width': '150px',
                        'backgroundColor': '#333333',
                        'color': 'white',
                        'marginRight': '10px'
                    }
                ),
                # Refresh Button
                html.Button(
                    '⟳ Refresh',
                    id='refresh-scanner',
                    style={
                        'backgroundColor': '#444444',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '8px 15px',
                        'cursor': 'pointer'
                    }
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginLeft': '20px'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '10px 5px',
            'backgroundColor': '#1a1a1a',
            'position': 'fixed',
            'top': '40px',
            'left': 0,
            'right': 0,
            'zIndex': 1000,
            'height': '50px'
        }),

        # Main Content
        html.Div([
            # Scanner Results Table
            html.Div([
                html.Div(id='scanner-results', style={
                    'backgroundColor': '#1a1a1a',
                    'borderRadius': '5px',
                    'padding': '20px',
                    'marginBottom': '20px'
                })
            ], style={
                'width': '70%',
                'padding': '20px'
            }),

            # Right Panel - Filters and Settings
            html.Div([
                # Filter Section
                html.Div([
                    html.H3('Filters', style={'color': 'white', 'marginBottom': '15px'}),
                    
                    html.Div([
                        html.Label('Price Range', style={'color': '#888', 'marginBottom': '5px'}),
                        dcc.RangeSlider(
                            id='price-range-slider',
                            min=0,
                            max=500,
                            step=1,
                            value=[0, 500],
                            marks={i: str(i) for i in range(0, 501, 100)}
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label('Volume Threshold', style={'color': '#888', 'marginBottom': '5px'}),
                        dcc.Input(
                            id='volume-threshold',
                            type='number',
                            placeholder='Min Volume',
                            style={
                                'width': '100%',
                                'backgroundColor': '#333333',
                                'color': 'white',
                                'border': '1px solid #666',
                                'borderRadius': '4px',
                                'padding': '5px'
                            }
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label('Market Cap', style={'color': '#888', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='market-cap-filter',
                            options=[
                                {'label': 'All', 'value': 'all'},
                                {'label': 'Large Cap', 'value': 'large'},
                                {'label': 'Mid Cap', 'value': 'mid'},
                                {'label': 'Small Cap', 'value': 'small'}
                            ],
                            value='all',
                            style={
                                'backgroundColor': '#333333',
                                'color': 'white'
                            }
                        )
                    ], style={'marginBottom': '20px'})
                ], style={
                    'backgroundColor': '#1a1a1a',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'marginBottom': '20px'
                }),

                # Scanner Settings
                html.Div([
                    html.H3('Scanner Settings', style={'color': 'white', 'marginBottom': '15px'}),
                    html.Div([
                        html.Label('Update Frequency', style={'color': '#888', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='update-frequency',
                            options=[
                                {'label': 'Real-time', 'value': 'realtime'},
                                {'label': '1 minute', 'value': '1min'},
                                {'label': '5 minutes', 'value': '5min'}
                            ],
                            value='1min',
                            style={
                                'backgroundColor': '#333333',
                                'color': 'white'
                            }
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label('Alert Settings', style={'color': '#888', 'marginBottom': '5px'}),
                        dcc.Checklist(
                            id='alert-settings',
                            options=[
                                {'label': ' Enable Desktop Notifications', 'value': 'desktop'},
                                {'label': ' Enable Sound Alerts', 'value': 'sound'}
                            ],
                            value=[],
                            style={'color': 'white'}
                        )
                    ])
                ], style={
                    'backgroundColor': '#1a1a1a',
                    'padding': '20px',
                    'borderRadius': '5px'
                })
            ], style={
                'width': '30%',
                'padding': '20px'
            })
        ], style={
            'display': 'flex',
            'marginTop': '90px',
            'backgroundColor': 'black'
        })
    ])


# === SECTION 6: MAIN APP LAYOUT ===
app.layout = html.Div([
    # Tab container with adjusted height and padding
    html.Div([
        dcc.Tabs(
            id='tabs',
            value='dashboard',
            style={
                'height': '40px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',  # Center the tabs
                'marginTop': '0px'
            },
            colors={
                'border': '#333',
                'primary': '#0d71df',
                'background': '#1a1a1a'
            },
            children=[
                dcc.Tab(
                    label='Dashboard',
                    value='dashboard',
                    style={
                        'backgroundColor': '#1a1a1a', 
                        'color': '#888', 
                        'height': '40px',
                        'lineHeight': '40px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'padding': '0 20px'
                    },
                    selected_style={
                        'backgroundColor': '#2a2a2a', 
                        'color': 'white', 
                        'height': '40px',
                        'lineHeight': '40px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'padding': '0 20px'
                    }
                ),
                dcc.Tab(
                    label='Analysis',
                    value='analysis',
                    style={
                        'backgroundColor': '#1a1a1a', 
                        'color': '#888', 
                        'height': '40px',
                        'lineHeight': '40px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'padding': '0 20px'
                    },
                    selected_style={
                        'backgroundColor': '#2a2a2a', 
                        'color': 'white', 
                        'height': '40px',
                        'lineHeight': '40px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'padding': '0 20px'
                    }
                ),
                dcc.Tab(
                    label='Scanner',
                    value='scanner',
                    style={
                        'backgroundColor': '#1a1a1a', 
                        'color': '#888', 
                        'height': '40px',
                        'lineHeight': '40px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'padding': '0 20px'
                    },
                    selected_style={
                        'backgroundColor': '#2a2a2a', 
                        'color': 'white', 
                        'height': '40px',
                        'lineHeight': '40px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'padding': '0 20px'
                    }
                )
            ]
        )
    ], style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'zIndex': 999,
        'backgroundColor': '#1a1a1a',
        'height': '40px',
        'borderBottom': '1px solid #333'
    }),
    
    dcc.Store(id='analysis-settings', storage_type='session'),

    dcc.Store(id='default-data-store'),
    html.Div(id='tab-content', style={
        'marginTop': '40px',  # Match the header height
        'backgroundColor': 'black',
        'minHeight': 'calc(100vh - 40px)',
        'overflowY': 'hidden',
        'overflowX': 'hidden'
    })
], style={
    'backgroundColor': 'black',
    'minHeight': '100vh',
    'width': '100vw',
    'margin': 0,
    'padding': 0,
    'position': 'relative',
    'overflowX': 'hidden'
})

# === SECTION 7: TAB CALLBACK ===
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'dashboard':
        return create_dashboard_layout()
    elif tab == 'analysis':
        return create_analysis_layout()
    elif tab == 'scanner':
        return create_scanner_layout()

@app.callback(
    Output('default-data-store', 'data'),
    Input('tabs', 'value')
)
def load_default_data(tab):
    if tab == 'analysis':
        symbol = 'SPY'
        timeframe = '1D'
        interval = '5min'
        df = get_data_for_timeframe_and_interval(symbol, timeframe, interval)
        if df is not None:
            # Convert datetime to string format before storing
            df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'interval': interval,
                'data': df.to_json(orient='split')
            }
    return None

# Global variables for historical price tracking
historical_prices = []
last_price_update = datetime.now()
data_cache = {}

def get_metric_color(value, metric_type):
    """
    Determine color based on fixed thresholds for each metric type.
    """
    if value == 0:
        return 'white'
    
    thresholds = {
        'GEX': {'low': 500000, 'medium': 1000000},
        'VEX': {'low': 200000, 'medium': 500000},
        'DEX': {'low': 1000000, 'medium': 2000000},
        'VEGA': {'low': 50000, 'medium': 100000}
    }
    
    threshold = thresholds.get(metric_type, thresholds['GEX'])
    
    if value > 0:
        if abs(value) <= threshold['low']:
            return '#FFA500'  # Orange
        elif abs(value) <= threshold['medium']:
            return '#FFFF00'  # Yellow
        else:
            return 'green'
    else:
        if abs(value) <= threshold['low']:
            return '#FFFF00'  # Yellow
        elif abs(value) <= threshold['medium']:
            return '#FFA500'  # Orange
        else:
            return 'red'

def get_tradier_credentials(is_sandbox):
    access_token = 'NEqtSeGLOhIM7yA0CdpFgcAw8KMv'
    base_url = 'https://sandbox.tradier.com/v1/' if is_sandbox else 'https://api.tradier.com/v1/'
    return access_token, base_url
def lookup_symbol(symbol, is_sandbox=False):
    """
    Look up a symbol to verify it exists and get basic information.
    
    Args:
        symbol (str): The symbol to look up (e.g., 'SPY')
        is_sandbox (bool): Whether to use sandbox environment
        
    Returns:
        bool: True if symbol exists and has options, False otherwise
    """
    access_token, base_url = get_tradier_credentials(is_sandbox)
    
    # First check if the symbol exists by getting a quote
    quote_url = f"{base_url}markets/quotes?symbols={symbol}"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    
    try:
        quote_response = requests.get(quote_url, headers=headers)
        if quote_response.status_code != 200:
            print(f"Symbol {symbol} not found")
            return False
            
        quote_data = quote_response.json()
        if 'quotes' not in quote_data or 'quote' not in quote_data['quotes']:
            print(f"No quote data for {symbol}")
            return False
            
        # Then check if it has options by getting expirations
        exp_url = f"{base_url}markets/options/expirations?symbol={symbol}"
        exp_response = requests.get(exp_url, headers=headers)
        
        if exp_response.status_code != 200:
            print(f"No options data for {symbol}")
            return False
            
        exp_data = exp_response.json()
        if 'expirations' in exp_data and exp_data['expirations'].get('date'):
            print(f"Symbol {symbol} validated successfully")
            return True
            
        print(f"No options expirations for {symbol}")
        return False
        
    except Exception as e:
        print(f"Error looking up symbol {symbol}: {str(e)}")
        return False

def validate_and_expand_symbols(symbols):
    """
    Validate a list of symbols and expand to include additional option roots.
    
    Args:
        symbols (list): List of underlying symbols to validate
        
    Returns:
        dict: Dictionary mapping base symbols to their option roots and available symbols
    """
    validated_symbols = {}
    
    for symbol in symbols:
        lookup_result = lookup_option_symbols(symbol)
        if lookup_result:
            validated_symbols[symbol] = lookup_result
            
    return validated_symbols


def load_custom_symbols():
    """Load custom symbols from file"""
    if os.path.exists(CUSTOM_SYMBOLS_FILE):
        try:
            with open(CUSTOM_SYMBOLS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_custom_symbols(symbols):
    """Save custom symbols to file"""
    with open(CUSTOM_SYMBOLS_FILE, 'w') as f:
        json.dump(symbols, f)

# Initialize with both default and custom symbols
def get_all_symbols():
    default_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
    custom_symbols = load_custom_symbols()
    return list(set(default_symbols + custom_symbols))  # Remove duplicates

@app.callback(
    [Output('date-dropdown', 'options'),
     Output('date-dropdown', 'value')],
    [Input('symbol-dropdown', 'value')]
)
def update_date_dropdown(symbol):
    try:
        expiration_dates = get_options_expirations(symbol)
        if not expiration_dates:
            # If no dates returned, create some default ones
            today = datetime.now()
            expiration_dates = [(today + timedelta(days=x)).strftime('%Y-%m-%d') 
                              for x in range(0, 28, 7)]
        
        options = [{'label': date, 'value': date} for date in expiration_dates]
        value = expiration_dates[0] if expiration_dates else None
        return options, value
    except Exception as e:
        print(f"Error in update_date_dropdown: {str(e)}")
        # Return default values
        today = datetime.now()
        default_dates = [(today + timedelta(days=x)).strftime('%Y-%m-%d') 
                        for x in range(0, 28, 7)]
        options = [{'label': date, 'value': date} for date in default_dates]
        return options, default_dates[0]

@app.callback(
    [Output('symbol-dropdown', 'options'),
     Output('symbol-input', 'value')],
    [Input('add-symbol-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('symbol-dropdown', 'options')]
)
def add_symbol(n_clicks, new_symbol, existing_options):
    if n_clicks is None or not new_symbol:
        # Initial load - combine default and custom symbols and sort
        all_symbols = sorted(get_all_symbols())
        return [{'label': s, 'value': s} for s in all_symbols], ''
    
    # Convert new symbol to uppercase
    new_symbol = new_symbol.strip().upper()
    
    # Validate the symbol using Tradier API
    if lookup_symbol(new_symbol):
        # Get existing symbols
        existing_symbols = [opt['value'] for opt in existing_options]
        
        if new_symbol not in existing_symbols:
            # Add to existing options
            existing_options.append({'label': new_symbol, 'value': new_symbol})
            
            # Save to custom symbols
            custom_symbols = load_custom_symbols()
            if new_symbol not in custom_symbols:
                custom_symbols.append(new_symbol)
                save_custom_symbols(custom_symbols)
            
            return existing_options, ''
        else:
            return existing_options, ''
    else:
        # If symbol is invalid, return unchanged options
        return existing_options, new_symbol

@app.callback(
    Output('symbol-input', 'style'),
    [Input('add-symbol-button', 'n_clicks')],
    [State('symbol-input', 'value')]
)
def update_input_style(n_clicks, symbol_value):
    base_style = {
        'width': '100px',
        'backgroundColor': '#333333',
        'color': 'white',
        'border': '1px solid #666',
        'borderRadius': '4px',
        'padding': '5px',
        'marginRight': '5px'
    }
    
    if n_clicks is None or not symbol_value:
        return base_style
    
    symbol_value = symbol_value.strip().upper()
    if lookup_symbol(symbol_value):
        return base_style
    else:
        base_style['border'] = '1px solid #ff4444'  # Red border for invalid symbols
        return base_style

def get_quote(symbols, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    symbols_str = ','.join(symbols)
    url = f"{base_url}markets/quotes?symbols={symbols_str}"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200 and 'quotes' in response.json():
        quotes = response.json()['quotes']
        if 'quote' in quotes:
            return quotes['quote']
    return None

def get_current_price(symbol, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    url = f"{base_url}markets/quotes?symbols={symbol}"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        quote_data = response.json().get('quotes', {}).get('quote', {})
        return quote_data.get('last')
    else:
        print(f"Failed to retrieve {symbol} quote data. Status code: {response.status_code}")
        return None

def is_trading_hours():
    """Check if market is currently open"""
    eastern_tz = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern_tz)
    
    # Check for weekends first
    if current_time.weekday() >= 5:  # Weekend
        return False
    
    # Define market hours in ET
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Check if within regular trading hours
    return market_open <= current_time <= market_close

def get_last_trading_day():
    """Get the most recent trading day with extended hours"""
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern)
    
    # If it's before market open (4 AM ET), use previous trading day
    if current_time.hour < 4:
        current_time = current_time - timedelta(days=1)
    
    # If it's weekend, adjust to Friday
    while current_time.weekday() >= 5:
        current_time = current_time - timedelta(days=1)
    
    return current_time.strftime('%Y-%m-%d')

def calculate_moving_averages(data, period=20):
    """Calculate EMA for the price data"""
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_sma(data, period=200):
    """Calculate SMA for the price data"""
    return data['close'].rolling(window=period).mean()

def create_colored_ma_optimized(df):
    """Create colors for all MAs using vectorized operations"""
    return {
        'ema20': np.where(df['close'] > df['EMA20'], '#26A69A', '#EF5350'),
        'sma200': np.where(df['close'] > df['SMA200'], '#26A69A', '#EF5350')
    }

def filter_trading_hours(df):
    """Filter data for regular trading hours only"""
    if df is None or df.empty:
        return df

    df = df.copy()
    
    # Ensure timezone is Pacific
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('US/Pacific')
    elif df['time'].dt.tz != pytz.timezone('US/Pacific'):
        df['time'] = df['time'].dt.tz_convert('US/Pacific')
    
    # Filter for trading days (Monday-Friday)
    df = df[df['time'].dt.dayofweek < 5]
    
    # Remove holidays
    holidays = get_market_holidays()
    df = df[~df['time'].dt.strftime('%Y-%m-%d').isin(holidays)]
    
    # Filter for regular trading hours (6:30 AM to 1:00 PM Pacific)
    mask = (df['time'].dt.time >= dt_time(6, 30)) & \
           (df['time'].dt.time <= dt_time(13, 0))
    df = df[mask]
    
    # Sort by time
    df = df.sort_values('time')
    
    # Create continuous time index by resetting timestamps
    df = df.copy()
    df['date'] = df['time'].dt.date
    dates = df['date'].unique()
    
    new_df = []
    base_time = df['time'].min()
    
    for i, date in enumerate(dates):
        day_data = df[df['date'] == date].copy()
        day_minutes = (day_data['time'].dt.hour * 60 + day_data['time'].dt.minute) - (6 * 60 + 30)
        day_data['time'] = base_time + pd.Timedelta(days=i) + pd.to_timedelta(day_minutes, unit='min')
        new_df.append(day_data)
    
    if new_df:
        df = pd.concat(new_df, ignore_index=True)
        df = df.drop('date', axis=1)
    
    return df    

def smooth_moving_averages(df):
    """
    Smooth moving averages across gaps in trading hours/days
    """
    if df is None or df.empty:
        return df
        
    try:
        # List of moving average columns
        ma_columns = ['EMA20', 'EMA50', 'SMA200']
        
        # Create a continuous time index at the data's frequency
        full_index = pd.date_range(
            start=df['time'].min(),
            end=df['time'].max(),
            freq=pd.infer_freq(df['time'])
        )
        
        # Reindex the dataframe with the continuous index
        df_continuous = df.set_index('time').reindex(full_index)
        
        # Forward fill the moving averages across gaps
        for col in ma_columns:
            if col in df_continuous.columns:
                df_continuous[col] = df_continuous[col].interpolate(method='linear')
        
        # Reset index and rename it back to 'time'
        df_continuous = df_continuous.reset_index()
        df_continuous = df_continuous.rename(columns={'index': 'time'})
        
        # Filter back to only trading hours/days
        df_continuous = filter_trading_hours(df_continuous)
        
        return df_continuous
        
    except Exception as e:
        print(f"Error in smooth_moving_averages: {str(e)}")
        return df

def process_dataframe(df, timeframe):
    if df is None or df.empty:
        return None
        
    try:
        df = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle datetime based on timeframe
        if timeframe == 'daily':
            df['time'] = pd.to_datetime(df['date'] if 'date' in df.columns else df['time'])
            df['time'] = df['time'].dt.tz_localize('US/Eastern', nonexistent='shift_forward').dt.tz_convert('US/Pacific')
            df = df[df['time'].dt.dayofweek < 5]  # Filter weekends
        else:
            df['time'] = pd.to_datetime(df['time'])
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('US/Eastern', nonexistent='shift_forward')
            df['time'] = df['time'].dt.tz_convert('US/Pacific')
            df['session'] = df['time'].apply(get_market_session)

        # Calculate technical indicators
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['SMA200'] = df['close'].rolling(window=200, min_periods=1).mean()
        
        # For longer timeframes, interpolate moving averages
        if timeframe in ['1W', '1M', '3M', '6M', '1Y']:
            for ma in ['EMA20', 'EMA50', 'SMA200']:
                df[ma] = df[ma].interpolate(method='linear')
        
        return df.ffill().bfill()  # Forward/backward fill NaN values
        
    except Exception as e:
        print(f"Error in process_dataframe: {str(e)}")
        traceback.print_exc()
        return None

def get_market_session(timestamp):
    """Determine market session (pre-market, regular, after-hours) in ET"""
    # Convert time to Eastern Time if it's not already
    if timestamp.tzinfo is None:
        eastern = pytz.timezone('US/Eastern')
        timestamp = eastern.localize(timestamp)
    elif timestamp.tzinfo != pytz.timezone('US/Eastern'):
        timestamp = timestamp.astimezone(pytz.timezone('US/Eastern'))
    
    hour = timestamp.hour
    minute = timestamp.minute
    
    # All times in ET
    if (hour < 4) or (hour == 4 and minute == 0):
        return None  # Before pre-market
    elif hour < 9 or (hour == 9 and minute < 30):
        return 'pre_market'  # 4:00 AM - 9:30 AM ET
    elif (hour == 9 and minute >= 30) or (hour > 9 and hour < 16):
        return 'regular'  # 9:30 AM - 4:00 PM ET
    elif hour < 20:
        return 'after_hours'  # 4:00 PM - 8:00 PM ET
    else:
        return None  # After post-market

def get_historical_prices_1min(symbol, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern)
    market_date = current_time.strftime('%Y-%m-%d')
    
    start_time = f"{market_date} 04:00:00"
    end_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    url = f"{base_url}markets/timesales?symbol={symbol}&interval=1min&start={start_time}&end={end_time}&session_filter=all"
    
    try:
        headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            response_json = response.json()
            if response_json and 'series' in response_json:
                history_data = response_json.get('series', {}).get('data', [])
                if history_data:
                    if isinstance(history_data, dict):
                        history_data = [history_data]
                    df = pd.DataFrame(history_data)
                    if not df.empty:
                        df['time'] = pd.to_datetime(df['time'])
                        df['time'] = df['time'].dt.tz_localize('US/Eastern')
                        df['session'] = df['time'].apply(get_market_session)
                        df['time'] = df['time'].dt.tz_convert('US/Pacific')
                        return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving 1-minute data: {str(e)}")
        return pd.DataFrame()

def get_historical_prices(symbol, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern)
    market_date = current_time.strftime('%Y-%m-%d')
    
    # Start from 6:30 AM ET
    start_time = f"{market_date} 06:30:00"
    
    # If current time is after 8 PM ET, use 8 PM as end time
    if current_time.hour >= 20:
        end_time = f"{market_date} 20:00:00"
    else:
        end_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    url = f"{base_url}markets/timesales"
    params = {
        'symbol': symbol,
        'interval': '5min',
        'start': start_time,
        'end': end_time,
        'session_filter': 'all'  # Get all session data
    }
    
    try:
        headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            response_json = response.json()
            if response_json and 'series' in response_json:
                history_data = response_json.get('series', {}).get('data', [])
                if history_data:
                    df = pd.DataFrame(history_data)
                    if not df.empty:
                        df['time'] = pd.to_datetime(df['time'])
                        df['time'] = df['time'].dt.tz_localize('US/Eastern')
                        return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving historical data: {str(e)}")
        return pd.DataFrame()

def get_data_for_timeframe_and_interval(symbol, timeframe, interval):
    try:
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Calculate number of calendar days to fetch based on timeframe
        days_to_fetch = {
            '1D': 2,      # Fetch 2 days to ensure we get full day
            '2D': 3,      # Fetch 3 days to ensure we get 2 days
            '3D': 4,      # Fetch 4 days to ensure we get 3 days
            '4D': 5,      # Fetch 5 days to ensure we get 4 days
            '1W': 8,      # Fetch 8 days to ensure we get a week
            '1M': 35,     # Fetch 35 days to ensure we get a month
            '3M': 95,     # Fetch 95 days to ensure we get 3 months
            '6M': 185,    # Fetch 185 days to ensure we get 6 months
            '1Y': 370     # Fetch 370 days to ensure we get a year
        }.get(timeframe, 2)
        
        start_time = now - timedelta(days=days_to_fetch)
        
        if interval == 'daily':
            df = get_daily_prices(symbol, start_time)
            if df is not None and not df.empty:
                df = process_daily_data(df, timeframe)
                return df
        else:
            interval_minutes = {
                '1min': 1,
                '5min': 5,
                '15min': 15,
                '30min': 30,
                '1hour': 60,
                '2hour': 120,
                '4hour': 240
            }.get(interval)
            
            if interval_minutes:
                df = get_timesales_data_improved(symbol, start_time, now, interval_minutes)
                if df is not None and not df.empty:
                    df = process_intraday_data(df, timeframe)
                    return df
        
        return None
        
    except Exception as e:
        print(f"Error getting data: {str(e)}")
        traceback.print_exc()
        return None

def process_daily_data(df, timeframe):
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Ensure time column exists and is timezone aware
    if 'time' not in df.columns:
        df['time'] = pd.to_datetime(df['date'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('US/Eastern')
    
    # Filter for trading days (Monday-Friday)
    df = df[df['time'].dt.dayofweek < 5]
    
    # Calculate indicators
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['SMA200'] = df['close'].rolling(window=200, min_periods=1).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df = calculate_ttm_squeeze(df)
    
    # Get the required number of trading days
    num_days = int(timeframe[0]) if timeframe[1] == 'D' else {
        '1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252
    }.get(timeframe, 1)
    
    # Sort by date and take the most recent N trading days
    df = df.sort_values('time', ascending=True)
    df = df.tail(num_days)
    
    # Convert time to end of day for proper display
    df['time'] = df['time'].apply(lambda x: x.replace(hour=16, minute=0))
    
    return df

def process_intraday_data(df, timeframe):
    """Process intraday data with proper trading day filtering"""
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Ensure timezone
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('US/Eastern')
    
    # Filter for trading days
    df = df[df['time'].dt.dayofweek < 5]
    
    # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
    trading_hours_mask = (
        ((df['time'].dt.hour == 9) & (df['time'].dt.minute >= 30)) |
        ((df['time'].dt.hour > 9) & (df['time'].dt.hour < 16))
    )
    df = df[trading_hours_mask]
    
    # Calculate indicators
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['SMA200'] = df['close'].rolling(window=200, min_periods=1).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df = calculate_ttm_squeeze(df)
    df['vwap'] = calculate_vwap(df)
    
    # Get unique trading days
    df['date'] = df['time'].dt.date
    trading_days = sorted(df['date'].unique())
    
    # Get required number of trading days
    num_days = int(timeframe[0]) if timeframe[1] == 'D' else {
        '1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252
    }.get(timeframe, 1)
    
    # Take the most recent N trading days
    if len(trading_days) > num_days:
        keep_days = trading_days[-num_days:]
        df = df[df['date'].isin(keep_days)]
    
    # Drop the date column we added
    df = df.drop('date', axis=1)
    
    return df

def filter_last_n_trading_days(df, n_days):
    """Filter dataframe to keep only the last N trading days"""
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Get unique trading days
    trading_days = pd.Series(df['time'].dt.date).unique()
    
    # Sort and take the last N days
    last_n_days = sorted(trading_days)[-n_days:]
    
    # Filter dataframe
    return df[df['time'].dt.date.isin(last_n_days)]

def get_timesales_data_improved(symbol, start_time, end_time, interval_minutes):
    """Get time sales data with improved error handling and chunking"""
    chunks = []
    current_start = start_time
    
    while current_start < end_time:
        chunk_end = min(current_start + timedelta(days=1), end_time)
        
        try:
            access_token, base_url = get_tradier_credentials(False)
            url = f"{base_url}markets/timesales"
            
            params = {
                'symbol': symbol,
                'interval': f'{interval_minutes}min',
                'start': current_start.strftime('%Y-%m-%d %H:%M:%S'),
                'end': chunk_end.strftime('%Y-%m-%d %H:%M:%S'),
                'session_filter': 'all'
            }
            
            headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                # Handle case where 'series' might be None
                if data and 'series' in data and data['series'] is not None:
                    # Handle case where 'data' might be None or not a list
                    series_data = data['series'].get('data', [])
                    if series_data:
                        # Convert single dict to list if necessary
                        if isinstance(series_data, dict):
                            series_data = [series_data]
                        chunk_df = pd.DataFrame(series_data)
                        if not chunk_df.empty:
                            chunks.append(chunk_df)
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching chunk for {symbol} from {current_start} to {chunk_end}: {str(e)}")
            traceback.print_exc()
        
        current_start = chunk_end + timedelta(seconds=1)
    
    if chunks:
        try:
            df = pd.concat(chunks, ignore_index=True)
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = df['time'].dt.tz_localize('US/Eastern')
            return df
        except Exception as e:
            print(f"Error processing chunks for {symbol}: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    print(f"No data found for {symbol} between {start_time} and {end_time}")
    return pd.DataFrame()

def resample_daily_data(df, rule):
    """Resample daily data to desired interval"""
    df = df.set_index('time')
    
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()

def get_daily_prices(symbol, start_date, is_sandbox=False):
    """Get daily price data with improved handling"""
    access_token, base_url = get_tradier_credentials(is_sandbox)
    
    if isinstance(start_date, (datetime, pd.Timestamp)):
        start_date = start_date.strftime('%Y-%m-%d')
    
    # Get current date in Eastern Time
    eastern_tz = pytz.timezone('US/Eastern')
    current_et = datetime.now(eastern_tz)
    end_date = current_et.strftime('%Y-%m-%d')
    
    url = f"{base_url}markets/history"
    params = {
        'symbol': symbol,
        'start': start_date,
        'end': end_date,
        'interval': 'daily'
    }
    
    try:
        headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and 'history' in data and data['history'] is not None and 'day' in data['history']:
                df = pd.DataFrame(data['history']['day'])
                if not df.empty:
                    df['time'] = pd.to_datetime(df['date'])
                    df['time'] = df['time'].dt.tz_localize('US/Eastern')
                    
                    # Convert to Pacific Time for display
                    df['time'] = df['time'].dt.tz_convert('US/Pacific')
                    
                    # Convert numeric columns
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df.sort_values('time')
        
        print(f"Failed to get daily data for {symbol}")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error retrieving daily data: {str(e)}")
        return pd.DataFrame()

def get_extended_strike_range(current_price, additional_strikes=3):
    """Calculate extended strike range"""
    strike_unit = 1.0  # Each strike is $1 apart
    base_strike = round(current_price)
    
    lower_strikes = [base_strike - (i * strike_unit) for i in range(1, additional_strikes + 1)]
    upper_strikes = [base_strike + (i * strike_unit) for i in range(1, additional_strikes + 1)]
    
    return min(lower_strikes), max(upper_strikes)

def get_options_expirations(symbol, is_sandbox=False, max_days=30):
    try:
        access_token, base_url = get_tradier_credentials(is_sandbox)
        url = f"{base_url}markets/options/expirations?symbol={symbol}"
        headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)  # Add timeout
            if response.status_code == 200:
                expiration_dates = response.json().get('expirations', {}).get('date', [])
                filtered_dates = []

                if not expiration_dates:
                    return []

                pst = pytz.timezone('America/Los_Angeles')
                now_pst = datetime.now(pst)

                for date in expiration_dates:
                    expiration_date = datetime.strptime(date, '%Y-%m-%d')
                    expiration_date_pst = pst.localize(expiration_date)
                    days_difference = (expiration_date_pst.date() - now_pst.date()).days

                    if 0 <= days_difference <= max_days:
                        filtered_dates.append(date)

                return filtered_dates
        except requests.exceptions.RequestException as e:
            print(f"Network error in get_options_expirations: {str(e)}")
            # Return some default dates as fallback
            today = datetime.now()
            default_dates = [(today + timedelta(days=x)).strftime('%Y-%m-%d') 
                           for x in range(0, max_days, 7)]
            return default_dates[:4]  # Return first 4 weekly expirations
            
    except Exception as e:
        print(f"Error in get_options_expirations: {str(e)}")
        return []

def get_options_data(symbol, expiration_date, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    url = f"{base_url}markets/options/chains?symbol={symbol}&expiration={expiration_date}&greeks=true"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        options_data = response.json()

        if options_data['options'] is None:
            print(f"No options data found for {symbol}, expiration {expiration_date}.")
            return None

        options_df = pd.DataFrame(options_data['options']['option'])
        current_price = get_current_price(symbol, is_sandbox)
        
        if current_price is None:
            print(f"Unable to get current price for {symbol}")
            return None
            
        # Tighter strike range
        lower_bound = current_price - 20
        upper_bound = current_price + 20
        options_df = options_df[(options_df['strike'] >= lower_bound) & (options_df['strike'] <= upper_bound)]

        return options_df
    else:
        print(f"Failed to retrieve options data for {symbol}, expiration {expiration_date}. Status code: {response.status_code}")
        return None

def calculate_iv(S, K, t, r, price, option_type):
    flag = 'c' if option_type == 'call' else 'p'
    implied_volatility = vectorized_implied_volatility(price, S, K, t, r, [flag], q=0, model='black', return_as='numpy')
    return implied_volatility[0]

def calculate_t(expiration_date):
    eastern_tz = pytz.timezone('US/Eastern')
    current_datetime = datetime.now(eastern_tz)
    expiration_datetime = datetime.strptime(expiration_date, '%Y-%m-%d').replace(tzinfo=eastern_tz)
    market_close_time = dt_time(20, 0)  # 4:00 PM EDT
    market_close_datetime = datetime.combine(expiration_datetime.date(), market_close_time).astimezone(eastern_tz)
    time_until_close = float((market_close_datetime - current_datetime).total_seconds()) // 60.0
    minutes_in_year = 365.25 * 24.00 * 60.00
    t = time_until_close / minutes_in_year if time_until_close is not None else None
    return t

def calculate_vanna(S, K, r, sigma, T):
    if sigma <= 0 or T <= 0:
        return 0
    try:
        d1 = (math.log(S / K) + (r - 0 + 0.5 * (sigma ** 2)) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        q = 0
        vanna = math.exp(-q * T) * norm.pdf(d1) * (d2 / sigma)
        return vanna
    except (ValueError, ZeroDivisionError):
        return 0

def calculate_greeks(options_df, S, r):
    if options_df is None or options_df.empty:
        return None
    for index, row in options_df.iterrows():
        K = row['strike']
        t = calculate_t(row['expiration_date'])
        try:
            iv = calculate_iv(S, K, t, r, row['last'], row['option_type'])
        except:
            iv = 0.3
        q = 0
        flag = 'c' if row['option_type'] == 'call' else 'p'
        try:
            greeks = get_all_greeks(flag, S, K, t, r, iv, q, model='black', return_as='dict')
            options_df.at[index, 't'] = t
            options_df.at[index, 'implied_volatility'] = iv
            options_df.at[index, 'delta'] = greeks['delta'][0]
            options_df.at[index, 'gamma'] = greeks['gamma'][0]
            options_df.at[index, 'theta'] = greeks['theta'][0]
            options_df.at[index, 'vega'] = greeks['vega'][0]
            options_df.at[index, 'rho'] = greeks['rho'][0]
            vanna = calculate_vanna(S, row['strike'], r, iv, t)
            options_df.at[index, 'vanna'] = vanna
        except:
            options_df.at[index, 't'] = t
            options_df.at[index, 'implied_volatility'] = 0.3
            options_df.at[index, 'delta'] = 0
            options_df.at[index, 'gamma'] = 0
            options_df.at[index, 'theta'] = 0
            options_df.at[index, 'vega'] = 0
            options_df.at[index, 'rho'] = 0
            options_df.at[index, 'vanna'] = 0
    return options_df

def determine_state(total_gex, total_vex):
    """Return state number and description"""
    print(f"Determining state - GEX: {total_gex}, VEX: {total_vex}")
    
    if total_gex > 0 and total_vex > 0:
        print("State 1: +GEX/+VEX (Balanced)")
        return 1, "+GEX/+VEX (Balanced - Price stabilizes between GEX levels)"
    elif total_gex > 0 and total_vex < 0:
        print("State 2: +GEX/-VEX (Trending)")
        return 2, "+GEX/-VEX (Trending - Momentum follows breakouts)"
    elif total_gex < 0 and total_vex > 0:
        print("State 3: -GEX/+VEX (Reversing)")
        return 3, "-GEX/+VEX (Reversing - Price gravitates to GEX strikes)"
    elif total_gex < 0 and total_vex < 0:
        print("State 4: -GEX/-VEX (Volatile)")
        return 4, "-GEX/-VEX (Volatile - Accelerated directional moves)"
    
    print("State 0: Undefined")
    return 0, "Undefined"

def calculate_flip_point(dataframe, current_price):
    """Calculate the zero gamma/flip point strike price"""
    try:
        # Group by strike and calculate weighted delta
        strikes_delta = dataframe.groupby('strike').apply(
            lambda x: (x['delta'] * x['open_interest']).sum() / x['open_interest'].sum()
        ).reset_index()
        strikes_delta.columns = ['strike', 'weighted_delta']
        
        # Find strike closest to 0.5 delta
        strikes_delta['delta_diff'] = abs(strikes_delta['weighted_delta'] - 0.5)
        flip_point = strikes_delta.loc[strikes_delta['delta_diff'].idxmin(), 'strike']
        
        return float(flip_point)
    except Exception as e:
        print(f"Error calculating flip point: {str(e)}")
        return None

def create_histogram(data, current_price, historical_prices, metric_type, skewness, left_strike, right_strike, height=400):
    positive_mask = data[metric_type] > 0
    negative_mask = data[metric_type] < 0
    
    max_positive = data[metric_type][positive_mask].max() if positive_mask.any() else None
    max_negative = data[metric_type][negative_mask].min() if negative_mask.any() else None
    
    fig = go.Figure(go.Bar(
        x=data['strike'],
        y=data[metric_type],
        marker=dict(
            color=np.where(data[metric_type] >= 0, '#13d133', '#f70a0a'),
            opacity=0.8
        ),
        width=0.8,
        name=metric_type
    ))
    
    # Add white borders for max values
    if max_positive is not None:
        max_pos_strike = data.loc[data[metric_type] == max_positive, 'strike'].iloc[0]
        fig.add_trace(go.Bar(
            x=[max_pos_strike],
            y=[max_positive],
            marker=dict(color='rgba(0,0,0,0)', line=dict(color='white', width=1)),
            width=0.79,
            offset=-0.395,
            showlegend=False
        ))
    
    if max_negative is not None:
        max_neg_strike = data.loc[data[metric_type] == max_negative, 'strike'].iloc[0]
        fig.add_trace(go.Bar(
            x=[max_neg_strike],
            y=[max_negative],
            marker=dict(color='rgba(0,0,0,0)', line=dict(color='white', width=1)),
            width=0.79,
            offset=-0.395,
            showlegend=False
        ))    

    fig.add_vline(
        x=current_price,
        line=dict(color="#0d71df", width=2, dash="dash"),
        annotation_text=f"{current_price:.2f}",
        annotation_position="top left"
    )

    if historical_prices:
        fig.add_trace(go.Scatter(
            x=historical_prices,
            y=[0] * len(historical_prices),
            mode='markers',
            marker=dict(color='yellow', size=5, symbol='circle'),
            name='Historical Prices',
            hovertemplate='Price: %{x:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        title={
            'text': f"{metric_type} at Each Strike",
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis=dict(
            range=[left_strike, right_strike],
            tickformat='.2f',
            gridcolor='#1e1e1e',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#1e1e1e',
            zeroline=True,
            zerolinecolor='#666666',
            zerolinewidth=1
        ),
        plot_bgcolor='rgb(0,0,0)',
        paper_bgcolor='rgb(0,0,0)',
        margin={'t': 30, 'b': 30, 'l': 30, 'r': 30}
    )
    return fig

def create_dex_histogram(data, current_price, historical_prices, left_strike, right_strike):
    positive_mask = data['DEX'] > 0
    negative_mask = data['DEX'] < 0
    
    max_positive = data['DEX'][positive_mask].max() if positive_mask.any() else None
    max_negative = data['DEX'][negative_mask].min() if negative_mask.any() else None
    
    fig = go.Figure(go.Bar(
        x=data['strike'],
        y=data['DEX'],
        marker=dict(
            color=np.where(data['DEX'] >= 0, '#13d133', '#f70a0a'),
            opacity=0.8
        ),
        width=0.8,
        showlegend=False,
        name='DEX'
    ))
    
    if max_positive is not None:
        max_pos_strike = data.loc[data['DEX'] == max_positive, 'strike'].iloc[0]
        fig.add_trace(go.Bar(
            x=[max_pos_strike],
            y=[max_positive],
            marker=dict(color='rgba(0,0,0,0)', line=dict(color='white', width=1)),
            width=0.79,
            offset=-0.395,
            showlegend=False
        ))
    
    if max_negative is not None:
        max_neg_strike = data.loc[data['DEX'] == max_negative, 'strike'].iloc[0]
        fig.add_trace(go.Bar(
            x=[max_neg_strike],
            y=[max_negative],
            marker=dict(color='rgba(0,0,0,0)', line=dict(color='white', width=1)),
            width=0.79,
            offset=-0.395,
            showlegend=False
        ))
    
    fig.add_vline(
        x=current_price,
        line=dict(color="#0d71df", width=2, dash="dash"),
        annotation_text=f"{current_price:.2f}",
        annotation_position="top left"
    )
    
    if historical_prices:
        fig.add_trace(go.Scatter(
            x=historical_prices,
            y=[0] * len(historical_prices),
            mode='markers',
            marker=dict(color='yellow', size=5, symbol='circle'),
            name='Historical Prices',
            showlegend=False
        ))
    
    fig.update_layout(
        template="plotly_dark",
        title={
            'text': "DEX at Each Strike",
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis=dict(
            range=[left_strike, right_strike],
            tickformat='.2f',
            gridcolor='#1e1e1e',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#1e1e1e',
            zeroline=True,
            zerolinecolor='#666666',
            zerolinewidth=1
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        margin={'t': 30, 'b': 30, 'l': 30, 'r': 30},
        height=300
    )
    return fig

def create_gauge(value, total_positive, total_negative, title):
    """
    Create a gauge chart with the specified metrics
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 45, 'color': "Green" if value > 50 else "Red"}},
        title={'text': title},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "Green",
                'tick0': 0,
                'dtick': 5
            },
            'bar': {'color': "Green" if value > 50 else "Red"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [{'range': [0, 100], 'color': '#111111'}],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 1,
                'value': value
            }
        },
        domain={'x': [0.1, 0.9], 'y': [0, 0.8]}
    ))
    
   #fig.add_annotation(
   #    x=0.74, y=0.25,
   #    text=f"Bullish<br>{total_positive / 1e6:.2f}M",
   #    showarrow=False,
   #    font={'color': "Green"}
   #)
   #fig.add_annotation(
   #    x=0.26, y=0.25,
   #    text=f"Bearish<br>{total_negative / 1e6:.2f}M",
   #    showarrow=False,
   #    font={'color': "Red"}
   #)
    
    fig.update_layout(
        height=300,
        margin={'t': 10, 'b': 10, 'l': 10, 'r': 10},
        template="plotly_dark",
        plot_bgcolor='black',
        paper_bgcolor='black'
    )
    return fig


def calculate_market_profile(df, num_bins=30):
    """
    Calculate market profile distribution and point of control
    
    Args:
        df: DataFrame with OHLCV data
        num_bins: Number of price levels for distribution
    
    Returns:
        tuple: (price levels, volume distribution, point of control)
    """
    try:
        # Create price bins using the range of prices
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / num_bins
        
        # Create bins for each price level
        bins = np.linspace(df['low'].min(), df['high'].max(), num_bins)
        
        # Initialize volume array for each price level
        volume_distribution = np.zeros(len(bins)-1)
        
        # Calculate volume distribution
        for i in range(len(df)):
            row = df.iloc[i]
            # Find which bins this candle spans
            low_idx = np.searchsorted(bins, row['low']) - 1
            high_idx = np.searchsorted(bins, row['high'])
            
            # Distribute volume across price levels
            if low_idx == high_idx:
                volume_distribution[low_idx] += row['volume']
            else:
                # Proportionally distribute volume across price levels
                span = high_idx - low_idx
                vol_per_level = row['volume'] / span
                volume_distribution[low_idx:high_idx] += vol_per_level
        
        # Find point of control (price level with highest volume)
        poc_idx = np.argmax(volume_distribution)
        point_of_control = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        return bins[:-1], volume_distribution, point_of_control
        
    except Exception as e:
        print(f"Error calculating market profile: {str(e)}")
        return None, None, None

def create_candlestick_figure(df, timeframe, gex_levels, current_price, symbol, flip_point=None):
    if df is None or df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Calculate market profile
    price_levels, volume_dist, poc = calculate_market_profile(df, num_bins=30)
    if price_levels is not None and volume_dist is not None:
        # Normalize volume distribution and add it as a horizontal bar chart
        normalized_volume = volume_dist / volume_dist.max() * 0.2
        
        fig.add_trace(go.Bar(
            x=normalized_volume,
            y=price_levels,
            orientation='h',
            name='Volume Profile',
            marker=dict(
                color='rgba(55, 128, 191, 0.1)',
                line=dict(color='rgba(55, 128, 191, 0.4)', width=1)
            ),
            showlegend=False,
            xaxis='x2'
        ))
        
        # Add Point of Control line if available
        if poc is not None:
            fig.add_hline(
                y=poc,
                line=dict(color='yellow', width=1, dash='dash')
            )
            
            fig.add_annotation(
                text=f"POC {poc:.2f}",
                x=1,
                y=poc,
                xref='paper',
                yref='y',
                showarrow=False,
                font=dict(size=10, color='yellow'),
                xshift=-10
            )

    # Add buffer to the time range
    if df is not None and not df.empty:
        last_time = df['time'].max()
        if timeframe == '1-minute':
            buffer = pd.Timedelta(minutes=60)
        elif timeframe == '5-minute':
            buffer = pd.Timedelta(minutes=65)
        elif timeframe == 'daily':
            buffer = pd.Timedelta(days=30)
        else:
            buffer = pd.Timedelta(minutes=0)
        
        x_range = [df['time'].min(), last_time + buffer]

    if timeframe != 'daily':
        df_copy = df.copy()
        
        if 'session' in df_copy.columns:
            df_copy = df_copy.sort_values('time').reset_index(drop=True)
            sessions = df_copy.groupby((df_copy['session'] != df_copy['session'].shift()).cumsum())
            
            for _, session_df in sessions:
                if session_df['session'].iloc[0] == 'pre_market':
                    fig.add_vrect(
                        x0=session_df['time'].min(),
                        x1=session_df['time'].max(),
                        fillcolor='gray',
                        opacity=0.2,
                        layer='below',
                        line_width=0
                    )
                elif session_df['session'].iloc[0] == 'after_hours':
                    fig.add_vrect(
                        x0=session_df['time'].min(),
                        x1=session_df['time'].max(),
                        fillcolor='gray',
                        opacity=0.2,
                        layer='below',
                        line_width=0
                    )

    if timeframe == 'daily':
        title_text = f"{symbol} Price (Daily)"
    else:
        chart_date = df['time'].dt.date.iloc[0]
        title_text = f"{symbol} Price ({timeframe}) {chart_date.strftime('%m/%d/%Y')}"

    if flip_point is not None and timeframe != 'daily':
        fig.add_hline(
            y=flip_point,
            line=dict(color='white', width=4, dash='solid'),
            annotation=dict(
                text=f"Flip Point: {flip_point:.2f}",
                font=dict(size=11, color='white'),
                xanchor='left',
                x=0.18
            )
        )

    fig.add_hline(
        y=current_price,
        line=dict(color='#0c0ce8', width=2, dash='dot')
    )

    price_color = '#26A69A' if df['close'].iloc[-1] > df['open'].iloc[-1] else '#EF5350'
    fig.add_annotation(
        text=f"{current_price:.2f}",
        xref='paper',
        x=1,
        yref='y',
        y=current_price,
        xshift=50,
        showarrow=False,
        font=dict(size=12, color=price_color)
    )

    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing=dict(
            line=dict(color='#07f256', width=1),
            fillcolor='#07f256'
        ),
        decreasing=dict(
            line=dict(color='#FF0000', width=1),
            fillcolor='#FF0000'
        ),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['EMA20'],
        line=dict(color='white', width=2, dash='solid'),
        name='20 EMA',
        connectgaps=True
    ))

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['SMA200'],
        line=dict(color='#FF00FF', width=4, dash='dot'),
        name='200 SMA',
        connectgaps=True
    ))

    if gex_levels and timeframe != 'daily':
        num_levels = 3 if timeframe == '1-minute' else 4
        pos_levels = [(level, value) for level, value in gex_levels if value > 0][:num_levels]
        neg_levels = [(level, value) for level, value in gex_levels if value < 0][:num_levels]
        
        price_range = max(df['high']) - min(df['low'])
        y_min = min(df['low']) - (price_range * 0.1)
        y_max = max(df['high']) + (price_range * 0.1)
        
        for i, (level, value) in enumerate(pos_levels, 1):
            line_width = 3 if i == 1 else 2
            dash_style = 'solid' if i == 1 else 'dash'
            fig.add_hline(
                y=level,
                line=dict(
                    color='#13d133',
                    width=line_width,
                    dash=dash_style
                ),
                annotation=dict(
                    text=f"+GEX {i}: {level:.2f}",
                    font=dict(size=10, color='#13d133'),
                    xanchor='left',
                    x=0.01
                )
            )
        
        for i, (level, value) in enumerate(neg_levels, 1):
            line_width = 3 if i == 1 else 2
            dash_style = 'solid' if i == 1 else 'dash'
            fig.add_hline(
                y=level,
                line=dict(
                    color='#f70a0a',
                    width=line_width,
                    dash=dash_style
                ),
                annotation=dict(
                    text=f"-GEX {i}: {level:.2f}",
                    font=dict(size=10, color='#f70a0a'),
                    xanchor='left',
                    x=0.01
                )
            )

        fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        title=dict(text=title_text, x=0.05, xanchor='left', yanchor='top'),
        template="plotly_dark",
        plot_bgcolor='black',
        paper_bgcolor='black',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.10,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            itemsizing='constant',
            orientation="h"
        ),
        height=450,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            range=x_range,
            rangeslider=dict(visible=False),
            type="date",
            gridcolor='#363535',
            uirevision='same',
            rangebreaks=None
        ),
        xaxis2=dict(
            overlaying='x',
            side='left',
            showgrid=False,
            showticklabels=False,
            range=[0, 0.3],
            domain=[0, 0.06],
            fixedrange=True
        ),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='#363535',
            uirevision='same'
        ),
        dragmode='pan'
    )
    return fig

def create_price_charts_optimized(historical_data_1min, historical_data_5min, historical_data_daily, gex_levels, current_price, symbol, day_change, flip_point=None):
    return html.Div([
        html.Div(
            dcc.Graph(
                id='chart-1min',
                figure=create_candlestick_figure(historical_data_1min, '1-minute', gex_levels, current_price, symbol, flip_point),
                config={'displayModeBar': False, 'scrollZoom': True}
            ),
            style={'marginBottom': '20px', 'border': '1px solid #333', 'borderRadius': '5px', 'backgroundColor': 'black'}
        ),
        html.Div(
            dcc.Graph(
                id='chart-5min',
                figure=create_candlestick_figure(historical_data_5min, '5-minute', gex_levels, current_price, symbol, flip_point),
                config={'displayModeBar': False, 'scrollZoom': True}
            ),
            style={'marginBottom': '20px', 'border': '1px solid #333', 'borderRadius': '5px', 'backgroundColor': 'black'}
        ),
        html.Div(
            dcc.Graph(
                id='chart-daily',
                figure=create_candlestick_figure(historical_data_daily, 'daily', gex_levels, current_price, symbol, flip_point),
                config={'displayModeBar': False, 'scrollZoom': True}
            ),
            style={'border': '1px solid #333', 'borderRadius': '5px', 'backgroundColor': 'black'}
        )
    ], style={'backgroundColor': 'black', 'width': '100%'})

def default_layout_values():
    default_fig = go.Figure(data=[],
        layout=go.Layout(
            template="plotly_dark",
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
    )
    
    default_output = [html.P(["No data available"], style={'color': 'white'})]
    default_state_style = {'color': 'white', 'opacity': '0.5'}
    
    return (
        default_output,      # output-container
        default_fig,         # dex-gauge-chart
        default_fig,         # gex-gauge-chart
        default_fig,         # vex-gauge-chart
        default_fig,         # gex-histogram
        default_fig,         # vex-histogram
        default_fig,         # dex-histogram
        html.Div("No data available", style={'color': 'white', 'text-align': 'center'}),  # price-charts
        default_state_style,  # state1-title
        default_state_style,  # state2-title
        default_state_style,  # state3-title
        default_state_style   # state4-title
    )

# =====================
# Metrics Calculations
# =====================
def get_metrics_by_strike(dataframe, current_price):
    """Calculate metrics by strike price"""
    try:
        gex_by_strike = dataframe.groupby('strike', group_keys=False).apply(
            lambda x: sum(
                x[x['option_type'] == 'call']['gamma'] * 
                x[x['option_type'] == 'call']['open_interest'] * 
                100 * current_price * current_price * 0.01
            ) + sum(
                x[x['option_type'] == 'put']['gamma'] * 
                x[x['option_type'] == 'put']['open_interest'] * 
                100 * current_price * current_price * 0.01 * -1
            )
        ).reset_index(name='GEX')
        
        vex_by_strike = dataframe.groupby('strike', group_keys=False).apply(
            lambda x: sum(
                x['vanna'] * x['open_interest'] * 
                x['implied_volatility'] * current_price
            )
        ).reset_index(name='VEX')
        
        dex_by_strike = dataframe.groupby('strike', group_keys=False).apply(
            lambda x: sum(
                x['delta'] * x['open_interest'] * 100
            )
        ).reset_index(name='DEX')
        
        gex_by_strike['color'] = np.where(gex_by_strike['GEX'] > 0, 'green', 'red')
        vex_by_strike['color'] = np.where(vex_by_strike['VEX'] > 0, 'green', 'red')
        dex_by_strike['color'] = np.where(dex_by_strike['DEX'] > 0, 'green', 'red')
        
        return gex_by_strike, vex_by_strike, dex_by_strike
        
    except Exception as e:
        print(f"Error in get_metrics_by_strike: {str(e)}")
        return pd.DataFrame({'strike': [], 'GEX': [], 'color': []}), pd.DataFrame({'strike': [], 'VEX': [], 'color': []}), pd.DataFrame({'strike': [], 'DEX': [], 'color': []})

def get_previous_close(symbol, is_sandbox=False):
    """Get the previous day's closing price"""
    access_token, base_url = get_tradier_credentials(is_sandbox)
    
    url = f"{base_url}markets/quotes?symbols={symbol}"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'quotes' in data and 'quote' in data['quotes']:
                return data['quotes']['quote'].get('prevclose')
    except Exception as e:
        print(f"Error getting previous close: {e}")
    return None

def get_advances_declines(is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    url = f"{base_url}markets/advances_declines"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['advances_declines']
        return None
    except Exception as e:
        print(f"Error getting advances/declines: {e}")
        return None

   
    # Determine which chart to adjust
    if button_id.endswith('1min'):
        current_range = fig1['layout']['yaxis']['range']
        fig1['layout']['yaxis']['range'] = [r + adjustment for r in current_range]
    elif button_id.endswith('5min'):
        current_range = fig5['layout']['yaxis']['range']
        fig5['layout']['yaxis']['range'] = [r + adjustment for r in current_range]
    elif button_id.endswith('daily'):
        current_range = fig_daily['layout']['yaxis']['range']
        fig_daily['layout']['yaxis']['range'] = [r + adjustment for r in current_range]
    
    return fig1, fig5, fig_daily

def create_ma_signals(df):
    """Generate moving average signals with TTM Squeeze status and VWAP"""
    try:
        if df is None or df.empty:
            return html.Div("No data available", style={'color': 'white'})
        
        signals = []
        
        # MA Signals
        if 'EMA20' in df.columns and 'EMA50' in df.columns:
            if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]:
                signals.append(html.Div("20 EMA above 50 EMA (Bullish)", style={'color': 'green'}))
            else:
                signals.append(html.Div("20 EMA below 50 EMA (Bearish)", style={'color': 'red'}))
        
        if 'EMA50' in df.columns and 'SMA200' in df.columns:
            if df['EMA50'].iloc[-1] > df['SMA200'].iloc[-1]:
                signals.append(html.Div("50 EMA above 200 SMA (Golden Cross)", style={'color': 'green'}))
            else:
                signals.append(html.Div("50 EMA below 200 SMA (Death Cross)", style={'color': 'red'}))
        
        # TTM Squeeze Status
        df = calculate_ttm_squeeze(df)
        squeeze_status, squeeze_bars = get_squeeze_status(df)
        
        squeeze_color = 'red' if squeeze_status == "IN SQUEEZE" else 'white'
        squeeze_text = f"{squeeze_status}"
        if squeeze_bars > 0:
            squeeze_text += f" ({squeeze_bars} bars)"
        
        signals.append(html.Div([
            html.Div("TTM Squeeze:", style={'color': '#888', 'marginTop': '10px'}),
            html.Div(squeeze_text, style={'color': squeeze_color})
        ]))
        
        # Add VWAP Status
        if 'vwap' in df.columns:
            current_price = df['close'].iloc[-1]
            vwap = df['vwap'].iloc[-1]
            
            vwap_color = 'green' if current_price > vwap else 'red'
            vwap_text = f"Above VWAP" if current_price > vwap else "Below VWAP"
            
            signals.append(html.Div([
                html.Div("VWAP Status:", style={'color': '#888', 'marginTop': '10px'}),
                html.Div(f"{vwap_text} (VWAP: {vwap:.2f})", style={'color': vwap_color})
            ]))
        
        return html.Div(signals)
        
    except Exception as e:
        print(f"Error in create_ma_signals: {str(e)}")
        return html.Div("Error calculating signals", style={'color': 'white'})

def create_oscillator_signals(df):
    """Generate oscillator signals"""
    signals = []
    
    # RSI signals
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi > 70:
        signals.append(html.Div(f"RSI Overbought ({current_rsi:.1f})", style={'color': 'red'}))
    elif current_rsi < 30:
        signals.append(html.Div(f"RSI Oversold ({current_rsi:.1f})", style={'color': 'green'}))
    else:
        signals.append(html.Div(f"RSI Neutral ({current_rsi:.1f})", style={'color': 'white'}))
    
    return html.Div(signals)

def calculate_pivot_points(df):
    """Calculate pivot points"""
    pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
    r1 = 2 * pivot - df['low'].iloc[-1]
    s1 = 2 * pivot - df['high'].iloc[-1]
    r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
    s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
    
    return html.Div([
        html.Div(f"R2: {r2:.2f}", style={'color': 'white'}),
        html.Div(f"R1: {r1:.2f}", style={'color': 'white'}),
        html.Div(f"Pivot: {pivot:.2f}", style={'color': 'white'}),
        html.Div(f"S1: {s1:.2f}", style={'color': 'white'}),
        html.Div(f"S2: {s2:.2f}", style={'color': 'white'})
    ])

def identify_support_levels(df):
    """Identify key support levels"""
    # Simple implementation using recent lows
    recent_lows = df['low'].nsmallest(3)
    
    return html.Div([
        html.Div(f"Support 1: {recent_lows.iloc[0]:.2f}", style={'color': 'green'}),
        html.Div(f"Support 2: {recent_lows.iloc[1]:.2f}", style={'color': 'green'}),
        html.Div(f"Support 3: {recent_lows.iloc[2]:.2f}", style={'color': 'green'})
    ])

def identify_resistance_levels(df):
    """Identify key resistance levels"""
    # Simple implementation using recent highs
    recent_highs = df['high'].nlargest(3)
    
    return html.Div([
        html.Div(f"Resistance 1: {recent_highs.iloc[0]:.2f}", style={'color': 'red'}),
        html.Div(f"Resistance 2: {recent_highs.iloc[1]:.2f}", style={'color': 'red'}),
        html.Div(f"Resistance 3: {recent_highs.iloc[2]:.2f}", style={'color': 'red'})
    ])

def calculate_iv_percentile(options_data):
    """Calculate IV percentile with better error handling"""
    try:
        if options_data is None or options_data.empty:
            return html.Div("No options data available", style={'color': 'white'})
        
        # Check if implied_volatility column exists
        if 'implied_volatility' not in options_data.columns:
            # Try to get IV from other possible column names
            iv_column = None
            possible_names = ['implied_volatility', 'iv', 'impliedVolatility']
            
            for name in possible_names:
                if name in options_data.columns:
                    iv_column = name
                    break
            
            if iv_column is None:
                return html.Div("Implied volatility data not available", style={'color': 'white'})
            
            current_iv = options_data[iv_column].mean()
            percentile = options_data[iv_column].rank(pct=True).mean() * 100
        else:
            current_iv = options_data['implied_volatility'].mean()
            percentile = options_data['implied_volatility'].rank(pct=True).mean() * 100
        
        # Handle NaN values
        if pd.isna(current_iv) or pd.isna(percentile):
            return html.Div("Unable to calculate IV metrics", style={'color': 'white'})
        
        return html.Div([
            html.Div(f"Current IV: {current_iv:.1f}%", style={'color': 'white'}),
            html.Div(f"IV Percentile: {percentile:.1f}%", style={'color': 'white'})
        ])
        
    except Exception as e:
        print(f"Error calculating IV percentile: {str(e)}")
        return html.Div("Error calculating IV metrics", style={'color': 'white'})

# Updated put/call ratio calculation with better error handling
def calculate_put_call_ratio(options_data):
    """Calculate put/call ratio with improved error handling"""
    try:
        if options_data is None or options_data.empty:
            return html.Div("No options data available", style={'color': 'white'})
        
        # Check if required columns exist
        if 'option_type' not in options_data.columns or 'volume' not in options_data.columns:
            return html.Div("Required options data not available", style={'color': 'white'})
        
        # Calculate volumes with error checking
        put_volume = options_data[options_data['option_type'] == 'put']['volume'].sum()
        call_volume = options_data[options_data['option_type'] == 'call']['volume'].sum()
        
        if call_volume == 0:
            return html.Div("No call volume data available", style={'color': 'white'})
        
        ratio = put_volume / call_volume
        
        return html.Div([
            html.Div(f"Put/Call Ratio: {ratio:.2f}", style={'color': 'white'}),
            html.Div(f"Put Volume: {put_volume:,}", style={'color': 'white'}),
            html.Div(f"Call Volume: {call_volume:,}", style={'color': 'white'})
        ])
        
    except Exception as e:
        print(f"Error calculating put/call ratio: {str(e)}")
        return html.Div("Error calculating options ratios", style={'color': 'white'})

@app.callback(
    [
        Output('output-container', 'children'),
        Output('header-price', 'children'),
        Output('dex-gauge-chart', 'figure'),
        Output('gex-gauge-chart', 'figure'),
        Output('vex-gauge-chart', 'figure'),
        Output('gex-histogram', 'figure'),
        Output('vex-histogram', 'figure'),
        Output('dex-histogram', 'figure'),
        Output('price-charts', 'children'),
        Output('state1-title', 'style'),
        Output('state2-title', 'style'),
        Output('state3-title', 'style'),
        Output('state4-title', 'style')
    ],
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('date-dropdown', 'value')]
)

def update_output(n_intervals, symbol, selected_date):
    global historical_prices, last_price_update, data_cache

    if not selected_date:
        return default_layout_values()
        
    try:
        # Add caching for frequent updates
        cache_key = f"{symbol}_{selected_date}"
        
        # Only update on interval if we're looking at live data
        if n_intervals and cache_key in data_cache:
            if (datetime.now() - data_cache[cache_key]['timestamp']).seconds >= 5:
                return data_cache[cache_key]['data']
        
        print(f"\n=== Starting data fetch for {symbol} ===")
        current_price = get_current_price(symbol)
        print(f"{symbol} Current Price: {current_price}")
        
        if current_price is None:
            print("Failed to get current price")
            return default_layout_values()
        
        prev_close = get_previous_close(symbol)
        print(f"Previous Close: {prev_close}")
        
        day_change = current_price - prev_close if prev_close is not None else 0
        print(f"Day Change: {day_change}")
        
        print("\nFetching options data...")
        expiration_dates = get_options_expirations(symbol)
        if not expiration_dates:
            print("No expiration dates found")
            return default_layout_values()
            
        options_df = get_options_data(symbol, selected_date)
        if options_df is None or options_df.empty:
            print("No options data found")
            return default_layout_values()
        
        print("\nFetching historical data...")
        historical_data_1min = get_historical_prices_1min(symbol)
        print(f"1min data shape: {historical_data_1min.shape if historical_data_1min is not None else None}")
        
        historical_data_5min = get_historical_prices(symbol)
        print(f"5min data shape: {historical_data_5min.shape if historical_data_5min is not None else None}")
        
        start_date = datetime.now() - timedelta(days=365)
        historical_data_daily = get_daily_prices(symbol, start_date)
        print(f"Daily data shape: {historical_data_daily.shape if historical_data_daily is not None else None}")
        
        print("\nProcessing dataframes...")
        df_1min = process_dataframe(historical_data_1min, '1-minute')
        print(f"Processed 1min shape: {df_1min.shape if df_1min is not None else None}")
        
        df_5min = process_dataframe(historical_data_5min, '5-minute')
        print(f"Processed 5min shape: {df_5min.shape if df_5min is not None else None}")
        
        df_daily = process_dataframe(historical_data_daily, 'daily')
        print(f"Processed daily shape: {df_daily.shape if df_daily is not None else None}")

        if df_1min is None and df_5min is None:
            print("Both processed dataframes are None!")
            return default_layout_values()
        
      
        # Calculate Greeks and metrics
        print("\nCalculating Greeks...")
        dataframe = calculate_greeks(options_df, current_price, .0548)
        if dataframe is None or dataframe.empty:
            print("Failed to calculate Greeks")
            return default_layout_values()

             
        # Calculate metrics
        # For metrics calculations
        gex_calls = dataframe[dataframe['option_type'] == 'call'].apply(
            lambda row: row['gamma'] * row['open_interest'] * 100 * current_price * current_price * 0.01, axis=1)
        gex_puts = dataframe[dataframe['option_type'] == 'put'].apply(
            lambda row: row['gamma'] * row['open_interest'] * 100 * current_price * current_price * 0.01 * -1, axis=1)

        vex_calls = dataframe[dataframe['option_type'] == 'call'].apply(
            lambda row: row['vanna'] * row['open_interest'] * row['implied_volatility'] * current_price, axis=1)
        vex_puts = dataframe[dataframe['option_type'] == 'put'].apply(
            lambda row: row['vanna'] * row['open_interest'] * row['implied_volatility'] * current_price, axis=1)     
            
        dex_calls = dataframe[dataframe['option_type'] == 'call'].apply(
            lambda row: row['delta'] * row['open_interest'] * 100, axis=1)
        dex_puts = dataframe[dataframe['option_type'] == 'put'].apply(
            lambda row: row['delta'] * row['open_interest'] * 100, axis=1)
        
        # Calculate metrics by strike
        gex_by_strike, vex_by_strike, dex_by_strike = get_metrics_by_strike(dataframe, current_price)
       
        # Calculate totals
        total_gex = gex_calls.sum() + gex_puts.sum()
        total_vex = vex_calls.sum() + vex_puts.sum()
        total_dex = dex_calls.sum() + dex_puts.sum()
       
        # Calculate Vega
        dataframe['Vega_Calls'] = dataframe.apply(
            lambda row: row['vega'] * row['open_interest'] * 100 if row['option_type'] == 'call' else 0, axis=1)
        dataframe['Vega_Puts'] = dataframe.apply(
            lambda row: row['vega'] * row['open_interest'] * -100 if row['option_type'] == 'put' else 0, axis=1)
        total_vega = dataframe['Vega_Calls'].sum() + dataframe['Vega_Puts'].sum()

        # Calculate flip point
        flip_point = calculate_flip_point(dataframe, current_price)
       
        # Determine market state
        current_state, state_description = determine_state(total_gex, total_vex)
       
        # Calculate percentages
        total_pct_Call_dex = abs(dex_calls.sum())
        total_pct_Put_dex = abs(dex_puts.sum())
        total_absolute_dex = total_pct_Call_dex + total_pct_Put_dex
        bullish_pct_dex = (total_pct_Call_dex / total_absolute_dex * 100) if total_absolute_dex != 0 else 0

        total_pct_Call_gex = abs(gex_calls.sum())
        total_pct_Put_gex = abs(gex_puts.sum())
        total_absolute_gex = total_pct_Call_gex + total_pct_Put_gex
        bullish_pct_gex = (total_pct_Call_gex / total_absolute_gex * 100) if total_absolute_gex != 0 else 0

        total_pct_Call_vex = abs(vex_calls.sum())
        total_pct_Put_vex = abs(vex_puts.sum())
        total_absolute_vex = total_pct_Call_vex + total_pct_Put_vex
        bullish_pct_vex = (total_pct_Call_vex / total_absolute_vex * 100) if total_absolute_vex != 0 else 0

        # Update historical prices
        current_time = datetime.now()
        if (current_time - last_price_update).total_seconds() >= 300:
            if current_price:
                historical_prices.append(current_price)
                last_price_update = current_time
                if len(historical_prices) > 80:
                    historical_prices.pop(0)
       
        # Get strike ranges
        if not gex_by_strike.empty:
            left_strike = min(current_price - 10, gex_by_strike['strike'].min())
            right_strike = max(current_price + 10, gex_by_strike['strike'].max())
        else:
            left_strike = current_price - 10
            right_strike = current_price + 10

        # Create histograms
        fig_gex_hist = create_histogram(gex_by_strike, current_price, historical_prices, 'GEX', 
                                      gex_by_strike['GEX'].skew(), left_strike, right_strike, height=250)
        fig_vex_hist = create_histogram(vex_by_strike, current_price, historical_prices, 'VEX', 
                                      vex_by_strike['VEX'].skew(), left_strike, right_strike, height=250)
        fig_dex_hist = create_dex_histogram(dex_by_strike, current_price, historical_prices, 
                                          left_strike, right_strike)

        # Get GEX levels for price charts
        gex_levels = []
        pos_gex = gex_by_strike[gex_by_strike['GEX'] > 0].nlargest(2, 'GEX')
        neg_gex = gex_by_strike[gex_by_strike['GEX'] < 0].nsmallest(2, 'GEX')

        if not pos_gex.empty:
            gex_levels.extend(list(zip(pos_gex['strike'], pos_gex['GEX'])))
        if not neg_gex.empty:
            gex_levels.extend(list(zip(neg_gex['strike'], neg_gex['GEX'])))

        # Create gauge figures
        fig_dex_gauge = create_gauge(bullish_pct_dex, total_pct_Call_dex, total_pct_Put_dex, "Bullish DEX vs Bearish DEX")
        fig_gex_gauge = create_gauge(bullish_pct_gex, total_pct_Call_gex, total_pct_Put_gex, "Bullish GEX vs Bearish GEX")
        fig_vex_gauge = create_gauge(bullish_pct_vex, total_pct_Call_vex, total_pct_Put_vex, "Bullish VEX vs Bearish VEX")

        # Create price charts
        price_charts = create_price_charts_optimized(df_1min, df_5min, df_daily, gex_levels, 
                                                   current_price, symbol, day_change, flip_point)

# Create state styles
        state_colors = {
            0: "#FFFFFF", 1: "#32CD32", 2: "#FFFF00", 3: "#FFA500", 4: "#FF0000"
        }
        
        state_styles = [
            {
                'color': state_colors[i],
                'font-weight': 'bold' if current_state == i else 'normal',
                'opacity': '1' if current_state == i else '0.5'
            }
            for i in range(1, 5)
        ]

        # Create tick matrix
        output_children = create_tick_matrix(total_gex, total_vex, total_dex, total_vega, 
                                           current_state, state_description, flip_point)

        # Create header price display
        header_price = html.Div([
            html.Span(f"{symbol}: ", style={'color': 'white'}),
            html.Span(f"{current_price:.2f} ", 
                     style={'color': '#32CD32' if day_change >= 0 else '#FF0000'}),
            html.Span(f"({'↑' if day_change >= 0 else '↓'}{abs(day_change):.2f})",
                     style={'color': '#32CD32' if day_change >= 0 else '#FF0000'})
        ])

        # Store the results for caching
        results = [
            output_children,
            header_price,
            fig_dex_gauge,
            fig_gex_gauge,
            fig_vex_gauge,
            fig_gex_hist,
            fig_vex_hist,
            fig_dex_hist,
            price_charts,
            state_styles[0],
            state_styles[1],
            state_styles[2],
            state_styles[3]
        ]

        # Cache the results
        cache_key = f"{symbol}_{selected_date}"
        data_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': results
        }

        return results

    except Exception as e:
        print(f"\n=== Error in update_output ===")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return [
            [],
            [],
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            html.Div("No data available", style={'color': 'white', 'text-align': 'center'}),
            {'color': 'white', 'opacity': '0.5'},
            {'color': 'white', 'opacity': '0.5'},
            {'color': 'white', 'opacity': '0.5'},
            {'color': 'white', 'opacity': '0.5'}
        ]


def determine_state(total_gex, total_vex):
    """Return state number and description"""
    print(f"Determining state - GEX: {total_gex}, VEX: {total_vex}")
    
    if total_gex > 0 and total_vex > 0:
        print("State 1: +GEX/+VEX (Balanced)")
        return 1, "+GEX/+VEX (Balanced - Price stabilizes between GEX levels)"
    elif total_gex > 0 and total_vex < 0:
        print("State 2: +GEX/-VEX (Trending)")
        return 2, "+GEX/-VEX (Trending - Momentum follows breakouts)"
    elif total_gex < 0 and total_vex > 0:
        print("State 3: -GEX/+VEX (Reversing)")
        return 3, "-GEX/+VEX (Reversing - Price gravitates to GEX strikes)"
    elif total_gex < 0 and total_vex < 0:
        print("State 4: -GEX/-VEX (Volatile)")
        return 4, "-GEX/-VEX (Volatile - Accelerated directional moves)"
    
    print("State 0: Undefined")
    return 0, "Undefined"

def create_state_section():
    return html.Div([
        # Market States Details
        html.Details([
            html.Summary('State 1: +GEX/+VEX (Balanced)', 
                        id='state1-title', 
                        style={'color': '#32CD32'}),
            html.Div([
                html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Upward moves trigger increasing selling pressure', style={'color': 'white'}),
                    html.Li('Downward moves trigger increasing buying pressure', style={'color': 'white'}),
                    html.Li('Price tends to stabilize between significant GEX levels', style={'color': 'white'})
                ]),
                html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Primary Setup: Identify largest green GEX bars as key levels', style={'color': 'white'}),
                    html.Li('Entry Points: Look for trades between major GEX strikes', style={'color': 'white'}),
                    html.Li('Price Below Flip Point: Expect bounce to first positive strike', style={'color': 'white'}),
                    html.Li('Best Opportunity: Range-bound trading between strong GEX levels', style={'color': 'white'})
                ])
            ], style={'padding': '10px'})
        ], style={'marginBottom': '10px'}),

        html.Details([
            html.Summary('State 2: +GEX/-VEX (Trending)', 
                        id='state2-title', 
                        style={'color': '#FFFF00'}),
            html.Div([
                html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Upward moves trigger increased buying pressure', style={'color': 'white'}),
                    html.Li('Downward moves trigger increased selling pressure', style={'color': 'white'}),
                    html.Li('Market tends toward trending movement', style={'color': 'white'})
                ]),
                html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Primary Setup: Monitor VEX chart for key levels', style={'color': 'white'}),
                    html.Li('Entry Points: Trade in direction of the break', style={'color': 'white'}),
                    html.Li('Risk Management: Use VEX levels as targets', style={'color': 'white'}),
                    html.Li('Best Opportunity: Momentum trades following breakouts', style={'color': 'white'})
                ])
            ], style={'padding': '10px'})
        ], style={'marginBottom': '10px'}),

        html.Details([
            html.Summary('State 3: -GEX/+VEX (Reversing)', 
                        id='state3-title', 
                        style={'color': '#FFA500'}),
            html.Div([
                html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Upward moves attract more sellers', style={'color': 'white'}),
                    html.Li('Price gravitates toward largest GEX strike', style={'color': 'white'}),
                    html.Li('Significant VEX levels influence price action', style={'color': 'white'})
                ]),
                html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Primary Setup: Short at closest positive GEX strike', style={'color': 'white'}),
                    html.Li('Entry Points: Look for price rejection at large VEX levels', style={'color': 'white'}),
                    html.Li('Risk Management: Use major GEX strikes as targets', style={'color': 'white'}),
                    html.Li('Best Opportunity: Short positions near significant resistance levels', style={'color': 'white'})
                ])
            ], style={'padding': '10px'})
        ], style={'marginBottom': '10px'}),

        html.Details([
            html.Summary('State 4: -GEX/-VEX (Volatile)', 
                        id='state4-title', 
                        style={'color': '#FF0000'}),
            html.Div([
                html.P('Key Characteristics:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Downward moves trigger accelerated selling pressure', style={'color': 'white'}),
                    html.Li('Upward bounces are typically weak', style={'color': 'white'}),
                    html.Li('Price tends to move rapidly until reaching major GEX level', style={'color': 'white'})
                ]),
                html.P('Trading Strategy:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Primary Setup: Look for confluence of large GEX and VEX levels', style={'color': 'white'}),
                    html.Li('Entry Points: Watch for price reaction at largest GEX strike levels', style={'color': 'white'}),
                    html.Li('Risk Management: Use VEX levels for potential support/resistance points', style={'color': 'white'}),
                    html.Li('Best Opportunity: Trade reversals when price reaches significant GEX levels', style={'color': 'white'})
                ])
            ], style={'padding': '10px'})
        ], style={'marginBottom': '10px'}),

        html.Details([
            html.Summary('Vega Summary', style={'color': 'white'}),
            html.Div([
                html.P('Market State Indicators:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Positive Total Vega (Long): Shows market positioning for volatility expansion', 
                           style={'color': 'white'}),
                    html.Li('Negative Total Vega (Short): Shows market positioning for volatility contraction', 
                           style={'color': 'white'}),
                    html.Li('VEX Chart: Maps key volatility exposure levels that can act as price inflection points', 
                           style={'color': 'white'})
                ]),
                html.P('Trading Applications:', style={'color': 'white', 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li('Large VEX strikes often act as significant price inflection points', 
                           style={'color': 'white'}),
                    html.Li('High positive VEX levels: Expect increased volatility and possible resistance', 
                           style={'color': 'white'}),
                    html.Li('High negative VEX levels: Expect decreased volatility and possible support', 
                           style={'color': 'white'}),
                    html.Li('Most reliable signals occur when major GEX and VEX levels align', 
                           style={'color': 'white'})
                ])
            ], style={'padding': '10px'})
        ])
    ], style={
        'backgroundColor': 'black',
        'padding': '20px',
        'marginTop': '40px'
    })

def create_tick_matrix(total_gex, total_vex, total_dex, total_vega, state_number, state_description, flip_point=None):
    return html.Div([
        # Options Flow Box (renamed from Tick Matrix)
        html.Div([
            html.Div("OPTIONS FLOW", style={
                'color': 'white',
                'fontSize': '12px',
                'fontWeight': 'bold',
                'borderBottom': '1px solid #333',
                'padding': '5px',
                'textAlign': 'center'  # Center the title
            }),
            html.Div([
                html.Div([
                    html.Span(f"State {state_number}: ", style={'color': 'white'}),
                    html.Span(state_description, 
                             style={'color': state_colors[state_number], 'fontWeight': 'bold'})
                ], style={'marginBottom': '5px'}),
                
                # Market Exposure Grid
                html.Div([
                    html.Div([
                         html.Div("FLIP POINT:", style={'color': '#888', 'width': '91px'}),
                         html.Div(f"${flip_point:.2f}", style={'color': 'white'})
                    ], style={'display': 'flex', 'marginBottom': '5px'}),
                           
                    html.Div([
                        html.Div("GEX:", style={'color': '#888', 'width': '40px'}),
                        html.Div(f"${int(total_gex):,}", 
                                style={'color': get_metric_color(total_gex, 'GEX')})
                    ], style={'display': 'flex', 'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Div("VEX:", style={'color': '#888', 'width': '40px'}),
                        html.Div(f"${int(total_vex):,}", 
                                style={'color': get_metric_color(total_vex, 'VEX')})
                    ], style={'display': 'flex', 'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Div("DEX:", style={'color': '#888', 'width': '40px'}),
                        html.Div(f"${int(total_dex):,}", 
                                style={'color': get_metric_color(total_dex, 'DEX')})
                    ], style={'display': 'flex', 'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Div("VEGA:", style={'color': '#888', 'width': '50px'}),
                        html.Div(f"${int(total_vega):,}", 
                                style={'color': get_metric_color(total_vega, 'VEGA')})
                    ], style={'display': 'flex', 'marginBottom': '5px'})
                ])
            ], style={'padding': '10px'})
        ], style={
            'backgroundColor': 'black',
            'border': '1px solid #333',
            'borderRadius': '5px',
            'marginBottom': '15px'
        }),
        
        # Market Pulse Box with Additional Metrics
        html.Div([
            html.Div("MARKET PULSE", style={
                'color': 'white',
                'fontSize': '12px',
                'fontWeight': 'bold',
                'borderBottom': '1px solid #333',
                'padding': '5px',
                'textAlign': 'center'  # Center the title
            }),
            html.Div([
                html.Div([
                    html.Div("Put/Call Ratio", style={'color': '#888', 'width': '120px'}),
                    html.Div("0.85", style={'color': 'white'})
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                
                html.Div([
                    html.Div("Vol Skew", style={'color': '#888', 'width': '120px'}),
                    html.Div("1.25", style={'color': 'white'})
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                
                html.Div([
                    html.Div("Net Delta", style={'color': '#888', 'width': '120px'}),
                    html.Div(f"{int(total_dex/1000)}K", 
                            style={'color': get_metric_color(total_dex, 'DEX')})
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                
                html.Div([
                    html.Div("Net Gamma", style={'color': '#888', 'width': '120px'}),
                    html.Div(f"{int(total_gex/1000)}K", 
                            style={'color': get_metric_color(total_gex, 'GEX')})
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                
                # Additional Market Pulse Metrics
                html.Div([
                    html.Div("IV Rank", style={'color': '#888', 'width': '120px'}),
                    html.Div("65%", style={'color': 'white'})  # You'll need to calculate this
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                
                html.Div([
                    html.Div("IV Percentile", style={'color': '#888', 'width': '120px'}),
                    html.Div("72%", style={'color': 'white'})  # You'll need to calculate this
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                
                html.Div([
                    html.Div("Option Volume", style={'color': '#888', 'width': '120px'}),
                    html.Div("125K", style={'color': 'white'})  # You'll need to calculate this
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                
                html.Div([
                    html.Div("Delta Skew", style={'color': '#888', 'width': '120px'}),
                    html.Div("0.92", style={'color': 'white'})  # You'll need to calculate this
                ], style={'display': 'flex', 'marginBottom': '5px'})
            ], style={'padding': '10px'})
        ], style={
            'backgroundColor': 'black',
            'border': '1px solid #333',
            'borderRadius': '5px',
            'marginBottom': '15px'
        })
    ], style={
        'padding': '15px',
        'backgroundColor': 'black',
        'marginTop': '5px'  # Reduced from 60px to move boxes up
    })

@app.callback(
    [Output('analysis-price-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('ma-signals', 'children'),
     Output('oscillator-signals', 'children'),
     Output('pivot-points', 'children'),
     Output('support-levels', 'children'),
     Output('resistance-levels', 'children'),
     Output('iv-percentile', 'children'),
     Output('put-call-ratio', 'children'),
     Output('analysis-header-price', 'children'),
     Output('analysis-settings', 'data')],
    [Input('analysis-interval-component', 'n_intervals'),
     Input('analysis-go-button', 'n_clicks')],
    [State('analysis-symbol-dropdown', 'value'),
     State('timeframe-dropdown', 'value'),
     State('interval-dropdown', 'value'),
     State('analysis-settings', 'data')],
    prevent_initial_call=True
)
def update_analysis_content(n_intervals, n_clicks, symbol, timeframe, interval, stored_settings):
    # Initialize with empty analysis
    empty_results = create_empty_analysis()
    
    try:
        # If no symbol or no trigger, return empty
        if not symbol:
            return empty_results
            
        # Use stored settings if available and no explicit inputs
        if n_clicks is None and n_intervals == 0:
            if stored_settings:
                symbol = stored_settings.get('symbol', 'SPY')
                timeframe = stored_settings.get('timeframe', '1D')
                interval = stored_settings.get('interval', '5min')
            else:
                symbol = 'SPY'
                timeframe = '1D'
                interval = '5min'
        
        # Get data
        df = get_data_for_timeframe_and_interval(symbol, timeframe, interval)
        if df is None or df.empty:
            return empty_results
        
        # Store current settings
        current_settings = {
            'symbol': symbol,
            'timeframe': timeframe,
            'interval': interval
        }
        
        # Get current price and calculate day change
        current_price = get_current_price(symbol)
        prev_close = get_previous_close(symbol)
        day_change = current_price - prev_close if all(x is not None for x in [current_price, prev_close]) else 0
        
        # Create header price display
        header_price = html.Div([
            html.Span(f"{symbol}: ", style={'color': 'white'}),
            html.Span(f"{current_price:.2f} ", style={'color': '#32CD32' if day_change >= 0 else '#FF0000'}),
            html.Span(f"({'↑' if day_change >= 0 else '↓'}{abs(day_change):.2f})", 
                     style={'color': '#32CD32' if day_change >= 0 else '#FF0000'})
        ])

        # Create all required outputs
        return (
            create_analysis_price_chart(df, symbol, timeframe),  # price chart
            create_rsi_chart(df, timeframe),                     # rsi chart with timeframe parameter
            create_ma_signals(df),                               # ma signals
            create_oscillator_signals(df),                       # oscillator signals
            calculate_pivot_points(df),                          # pivot points
            identify_support_levels(df),                         # support levels
            identify_resistance_levels(df),                      # resistance levels
            calculate_iv_percentile(get_options_data(symbol, datetime.now().strftime('%Y-%m-%d'))),  # iv percentile
            calculate_put_call_ratio(get_options_data(symbol, datetime.now().strftime('%Y-%m-%d'))), # put/call ratio
            header_price,                                        # header price
            current_settings                                     # settings
        )
        
    except Exception as e:
        print(f"Error in update_analysis_content: {str(e)}")
        traceback.print_exc()
        return empty_results

@app.callback(
    Output('analysis-symbol-dropdown', 'options'),
    [Input('tabs', 'value')],
    prevent_initial_call=True
)
def update_analysis_symbols(tab):
    if tab == 'analysis':
        all_symbols = sorted(get_all_symbols())  # Sort the symbols
        return [{'label': s, 'value': s} for s in all_symbols]
    return []

# Keep the existing callback for initializing analysis settings
@app.callback(
    [Output('analysis-symbol-dropdown', 'value'),
     Output('timeframe-dropdown', 'value'),
     Output('interval-dropdown', 'value')],
    [Input('tabs', 'value')],
    [State('analysis-settings', 'data')]
)
def initialize_analysis_settings(tab, stored_settings):
    all_symbols = get_all_symbols()  # Get sorted symbols
    if tab == 'analysis' and stored_settings:
        return (
            stored_settings.get('symbol', all_symbols[0]),
            stored_settings.get('timeframe', '1D'),
            stored_settings.get('interval', '5min')
        )
    return all_symbols[0], '1D', '5min'

def create_empty_figure():
    """Create an empty figure with dark theme"""
    return go.Figure(layout=dict(
        template="plotly_dark",
        plot_bgcolor='black',
        paper_bgcolor='black',
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor='#1f1f1f'),
        yaxis=dict(showgrid=True, gridcolor='#1f1f1f')
    ))

def create_empty_analysis():
    """Create empty outputs for the analysis dashboard."""
    empty_fig = create_empty_figure()
    empty_div = html.Div("No data available", style={'color': 'white'})
    empty_settings = {'symbol': 'SPY', 'timeframe': '1D', 'interval': '5min'}
    
    return (
        empty_fig,        # analysis-price-chart.figure
        empty_fig,        # rsi-chart.figure
        empty_div,        # ma-signals.children
        empty_div,        # oscillator-signals.children
        empty_div,        # pivot-points.children
        empty_div,        # support-levels.children
        empty_div,        # resistance-levels.children
        empty_div,        # iv-percentile.children
        empty_div,        # put-call-ratio.children
        empty_div,        # analysis-header-price.children
        empty_settings    # analysis-settings.data
    )

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator with improved error handling"""
    try:
        # Convert input to pandas Series if it isn't already
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Convert to float and handle NaN values
        prices = pd.to_numeric(prices, errors='coerce')
        prices = prices.dropna()
        
        # Return empty series if not enough data
        if len(prices) < period + 1:
            return pd.Series(index=prices.index)
        
        # Calculate price changes
        changes = prices.diff()
        
        # Create gains and losses series
        gains = changes.copy()
        losses = changes.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Fill NaN values with neutral RSI
        rsi = rsi.replace([np.inf, -np.inf], 50)  # Handle infinity cases
        
        return rsi
        
    except Exception as e:
        print(f"Error calculating RSI: {str(e)}")
        print(f"Input type: {type(prices)}")
        print(f"Input data head: {prices.head() if hasattr(prices, 'head') else prices}")
        # Return neutral RSI values instead of empty series
        return pd.Series([50] * len(prices), index=prices.index)

def analyze_rsi(rsi_values):
    """Analyze RSI values and return signals"""
    try:
        current_rsi = rsi_values.iloc[-1]
        signals = []
        
        if current_rsi > 70:
            signals.append({
                'condition': 'Overbought',
                'value': current_rsi,
                'color': 'red'
            })
        elif current_rsi < 30:
            signals.append({
                'condition': 'Oversold',
                'value': current_rsi,
                'color': 'green'
            })
        else:
            signals.append({
                'condition': 'Neutral',
                'value': current_rsi,
                'color': 'white'
            })
            
        return signals
    except Exception as e:
        print(f"Error analyzing RSI: {str(e)}")
        return [{'condition': 'Error', 'value': 0, 'color': 'white'}]

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    try:
        df = df.copy()
        
        # Calculate EMAs and SMAs
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['SMA200'] = df['close'].rolling(window=200, min_periods=1).mean()
        
        # Calculate RSI
        df['RSI'] = calculate_rsi(df['close'])
        
        # Calculate Bollinger Bands
        bb_period = 20
        df['Middle_BB'] = df['close'].rolling(window=bb_period).mean()
        rolling_std = df['close'].rolling(window=bb_period).std()
        df['Upper_BB'] = df['Middle_BB'] + (2 * rolling_std)
        df['Lower_BB'] = df['Middle_BB'] - (2 * rolling_std)
        
        return df.ffill().bfill()  # Forward/backward fill NaN values
        
    except Exception as e:
        print(f"Error in calculate_technical_indicators: {str(e)}")
        return df
        
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        traceback.print_exc()
        return df  # Return original dataframe if calculations fail


def create_oscillator_signals(df):
    """Generate oscillator signals including RSI"""
    if 'RSI' not in df.columns or df.empty:
        return html.Div("No RSI data available", style={'color': 'white'})
    
    try:
        rsi_signals = analyze_rsi(df['RSI'])
        signals = []
        
        for signal in rsi_signals:
            signals.append(html.Div(
                f"RSI {signal['condition']} ({signal['value']:.1f})",
                style={'color': signal['color']}
            ))
        
        return html.Div(signals)
        
    except Exception as e:
        print(f"Error creating oscillator signals: {str(e)}")
        return html.Div("Error calculating signals", style={'color': 'white'})

def create_empty_figure():
    """Create an empty figure with dark theme"""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='black',
        paper_bgcolor='black',
        showlegend=False,
        height=300,  # Set a default height
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor='#1f1f1f'),
        yaxis=dict(showgrid=True, gridcolor='#1f1f1f')
    )
    return fig

def create_volume_profile(df):
    """Create volume profile visualization"""
    try:
        fig = go.Figure()
        
        if not df.empty and 'close' in df.columns and 'volume' in df.columns:
            # Add small random noise to avoid duplicate edges
            jittered_prices = df['close'] + np.random.normal(0, 0.01, len(df['close']))
            
            # Create bins and calculate volume profile - now with observed=True
            try:
                price_bins = pd.qcut(jittered_prices, q=20, duplicates='drop')
                volume_profile = df.groupby(price_bins, observed=True)['volume'].sum()
                
                # Handle the case where bins might be empty
                if not volume_profile.empty:
                    fig.add_trace(go.Bar(
                        x=volume_profile.values,
                        y=[p.mid for p in volume_profile.index],
                        orientation='h',
                        name='Volume Profile',
                        marker_color='rgba(55, 128, 191, 0.7)'
                    ))
            except Exception as e:
                print(f"Error in volume profile calculation: {str(e)}")
        
        fig.update_layout(
            template="plotly_dark",
            title="Volume Profile",
            xaxis_title="Volume",
            yaxis_title="Price",
            height=300,
            plot_bgcolor='black',
            paper_bgcolor='black',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating volume profile: {str(e)}")
        return create_empty_figure()



def get_historical_prices_extended(symbol, start_date, interval='daily', is_sandbox=False):
    """Get extended historical price data"""
    access_token, base_url = get_tradier_credentials(is_sandbox)
    
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Format the request URL
    url = f"{base_url}markets/timesales"
    params = {
        'symbol': symbol,
        'interval': interval,
        'start': start_date.strftime('%Y-%m-%d'),
        'session_filter': 'all'
    }
    
    try:
        headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            response_json = response.json()
            if 'series' in response_json and 'data' in response_json['series']:
                data = response_json['series']['data']
                if data:
                    df = pd.DataFrame(data)
                    df['time'] = pd.to_datetime(df['time'])
                    df['time'] = df['time'].dt.tz_localize('US/Eastern')
                    df['session'] = df['time'].apply(get_market_session)
                    df['time'] = df['time'].dt.tz_convert('US/Pacific')
                    return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving extended historical data: {str(e)}")
        return pd.DataFrame()

def calculate_ttm_squeeze(df, bb_length=20, kc_length=20, bb_std=2, kc_std=1.5):
    """Calculate TTM Squeeze indicator with updated methods"""
    try:
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=bb_length).mean()
        rolling_std = df['close'].rolling(window=bb_length).std()
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * bb_std)
        
        # Calculate True Range correctly
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['TR'] = ranges.max(axis=1)
        
        # Calculate Keltner Channels
        df['ATR'] = df['TR'].rolling(window=kc_length).mean()
        df['KC_Middle'] = df['close'].rolling(window=kc_length).mean()
        df['KC_Upper'] = df['KC_Middle'] + (df['ATR'] * kc_std)
        df['KC_Lower'] = df['KC_Middle'] - (df['ATR'] * kc_std)
        
        # Calculate Squeeze
        df['Squeeze'] = (df['BB_Upper'] <= df['KC_Upper']) & (df['BB_Lower'] >= df['KC_Lower'])
        
        # Use ffill() instead of fillna(method='ffill')
        df['Squeeze'] = df['Squeeze'].ffill()
        
        return df
        
    except Exception as e:
        print(f"Error calculating TTM Squeeze: {str(e)}")
        traceback.print_exc()
        return df

def calculate_market_profile(df, num_bins=50):
    """
    Calculate market profile distribution and point of control
    
    Args:
        df: DataFrame with OHLCV data
        num_bins: Number of price levels for distribution
    
    Returns:
        tuple: (price levels, volume distribution, point of control)
    """
    try:
        # Create price bins using the range of prices
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / num_bins
        
        # Create bins for each price level
        bins = np.linspace(df['low'].min(), df['high'].max(), num_bins)
        
        # Initialize volume array for each price level
        volume_distribution = np.zeros(len(bins)-1)
        
        # Calculate volume distribution
        for i in range(len(df)):
            row = df.iloc[i]
            # Find which bins this candle spans
            low_idx = np.searchsorted(bins, row['low']) - 1
            high_idx = np.searchsorted(bins, row['high'])
            
            # Distribute volume across price levels
            if low_idx == high_idx:
                volume_distribution[low_idx] += row['volume']
            else:
                # Proportionally distribute volume across price levels
                span = high_idx - low_idx
                vol_per_level = row['volume'] / span
                volume_distribution[low_idx:high_idx] += vol_per_level
        
        # Find point of control (price level with highest volume)
        poc_idx = np.argmax(volume_distribution)
        point_of_control = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        return bins[:-1], volume_distribution, point_of_control
        
    except Exception as e:
        print(f"Error calculating market profile: {str(e)}")
        return None, None, None

def update_price_chart_with_squeeze(fig, df):
    """
    Add TTM Squeeze indicator to the price chart
    """
    # Calculate y-position for squeeze dots (below the candlesticks)
    y_min = df['low'].min()
    y_range = df['high'].max() - y_min
    squeeze_y = y_min - (y_range * 0.02)  # Place dots 2% below the lowest price
    
    # Add squeeze dots
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=[squeeze_y] * len(df),
        mode='markers',
        marker=dict(
            size=8,
            color=np.where(df['Squeeze'], 'red', 'white'),
            symbol='circle'
        ),
        name='TTM Squeeze',
        showlegend=True,
        hovertemplate='Squeeze: %{text}<extra></extra>',
        text=np.where(df['Squeeze'], 'Squeeze ON', 'Squeeze OFF')
    ))
    
    # Update y-axis range to accommodate squeeze dots
    fig.update_layout(
        yaxis=dict(
            range=[squeeze_y - (y_range * 0.01), df['high'].max() + (y_range * 0.01)]
        )
    )
    
    return fig

def filter_regular_trading_hours(df):
    """Filter dataframe for regular trading hours and create continuous timeline"""
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Convert to Eastern Time if not already
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('US/Pacific').dt.tz_convert('US/Eastern')
    elif df['time'].dt.tz != pytz.timezone('US/Eastern'):
        df['time'] = df['time'].dt.tz_convert('US/Eastern')
    
    # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
    df = df[
        ((df['time'].dt.hour == 9) & (df['time'].dt.minute >= 30)) |
        ((df['time'].dt.hour > 9) & (df['time'].dt.hour < 16))
    ]
    
    # Sort by time
    df = df.sort_values('time')
    
    # Create continuous timeline
    df['date'] = df['time'].dt.date
    dates = df['date'].unique()
    
    # Create new continuous timeline
    new_df = []
    base_time = df['time'].min()
    
    for i, date in enumerate(dates):
        day_data = df[df['date'] == date].copy()
        
        # Calculate minutes since market open for each data point
        market_open = pd.Timestamp(date).replace(hour=9, minute=30, tzinfo=pytz.timezone('US/Eastern'))
        minutes_from_open = (day_data['time'] - market_open).dt.total_seconds() / 60
        
        # Create new continuous timestamps
        day_data['time'] = base_time + pd.Timedelta(days=i) + pd.to_timedelta(minutes_from_open, unit='min')
        new_df.append(day_data)
    
    if new_df:
        df = pd.concat(new_df, ignore_index=True)
        df = df.drop('date', axis=1)
        
        # Convert back to Pacific Time for display
        df['time'] = df['time'].dt.tz_convert('US/Pacific')
    
    return df

def create_analysis_price_chart(df, symbol, timeframe):
    """Create price chart with VWAP for analysis tab"""
    if df is None or df.empty:
        return create_empty_figure()

    try:
        # Create continuous timeline for non-daily timeframes
        if timeframe != '1Y':
            df = create_continuous_timeline(df)
            
            # Calculate VWAP for non-daily timeframes
            df['vwap'] = calculate_vwap(df)
        
        if df.empty:  # Check again after filtering
            return create_empty_figure()
        
        # Get current price early
        current_price = get_current_price(symbol)
        if current_price is None:
            current_price = df['close'].iloc[-1]
        
        # Create continuous timeline without gaps
        if timeframe != '1Y':
            df = create_continuous_timeline(df)
        
        # Calculate time ranges for x-axis
        current_time = df['time'].iloc[-1]
        time_range = current_time - df['time'].min()
        half_range = time_range / 2
        initial_range = [current_time - half_range, current_time + timedelta(hours=2)]
    
        # Calculate price ranges
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        price_margin = price_range * 0.2
    
        # Calculate max volume once
        volume_max = df['volume'].max()

        # Calculate Supply and Demand Visible Range
        visible_range = VisibleRange(threshold_percent=10, resolution=50)
        visible_range.calculate_zones(df)
        
        # Create figure with secondary y-axis for volume
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add volume bars
        green_volume_mask = df['close'] >= df['open']
        red_volume_mask = df['close'] < df['open']
        
        # Add green volume bars
        fig.add_trace(
            go.Bar(
                x=df[green_volume_mask]['time'],
                y=df[green_volume_mask]['volume'],
                marker_color='#07f256',
                opacity=0.3,
                name='Volume',
                showlegend=False,
                hovertemplate='Volume: %{customdata:,.0f}<extra></extra>',
                customdata=df[green_volume_mask]['volume']
            ),
            secondary_y=False
        )

        # Add red volume bars
        fig.add_trace(
            go.Bar(
                x=df[red_volume_mask]['time'],
                y=df[red_volume_mask]['volume'],
                marker_color='#FF0000',
                opacity=0.3,
                name='Volume',
                showlegend=False,
                hovertemplate='Volume: %{customdata:,.0f}<extra></extra>',
                customdata=df[red_volume_mask]['volume']
            ),
            secondary_y=False
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing=dict(line=dict(color='#07f256', width=1), fillcolor='#07f256'),
                decreasing=dict(line=dict(color='#FF0000', width=1), fillcolor='#FF0000'),
                name='Price',
                showlegend=False,
                text=[f"{t.strftime('%a %b %d, %Y, %H:%M')}<br>{c:.2f}" 
                      for t, c in zip(df['time'], df['close'])],
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='rgba(0,0,0,0.8)',
                    namelength=0
                )
            ),
            secondary_y=True
        )

        # Add TTM Squeeze dots if available
        if 'Squeeze' in df.columns:
            squeeze_points_x = df[df['Squeeze']]['time']
            if not squeeze_points_x.empty:
                fig.add_trace(
                    go.Scatter(
                        x=squeeze_points_x,
                        y=[volume_max * 1.9] * len(squeeze_points_x),
                        mode='markers',
                        marker=dict(size=5, color='red', symbol='circle'),
                        name='TTM Squeeze',
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=False
                )

        # Add current price line and enhanced box annotation
        if current_price:
            fig.add_hline(
                y=current_price,
                line=dict(color='#0c0ce8', width=2, dash='dot'),
                secondary_y=True
            )
            
            # Add current price annotation with box
            price_color = '#07f256' if df['close'].iloc[-1] > df['open'].iloc[-1] else '#FF0000'
            fig.add_annotation(
                text=f"{current_price:.2f}",
                xref='paper',
                x=1,
                yref='y2',
                y=current_price,
                xshift=40,
                showarrow=False,
                font=dict(size=14, color=price_color),
                bgcolor='black',
                bordercolor='white',
                borderwidth=1,
                borderpad=4,
                opacity=1
            )

        # Add supply and demand zones
        visible_range.add_zones_to_chart(fig, df)

        # Update layout with proper axes configuration
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"{symbol} Price ({timeframe}) {datetime.now().strftime('%m/%d/%Y')}" if timeframe != 'daily' else f"{symbol} Price (Daily)",
                x=0.05,
                font=dict(size=14)
            ),
            xaxis=dict(
                rangeslider=dict(visible=False),
                type="date",
                showspikes=True,
                spikesnap='cursor',
                spikemode='across',
                spikethickness=1,
                spikecolor='#999999',
                showline=True,
                showgrid=True,
                gridcolor='#1f1f1f',
                domain=[0.05, 1],
                range=initial_range,
                autorange=False,            
                uirevision='shared_x_range',           
                tickmode='auto',
                tickformat='%H:%M' if timeframe in ['1D', '2D', '3D', '4D'] else '%b %d',
                                          dtick='M30' if timeframe in ['1D', '2D', '3D', '4D'] else 'D1',
                                          hoverformat='%Y-%m-%d'
                
            ),
            height=745,
            plot_bgcolor='black',
            paper_bgcolor='black',
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                yanchor="top",
                y=1.1,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10),
                orientation="h"
            ),
            barmode='overlay',
            hovermode='x',
            hoverdistance=100,
            dragmode='pan'
        )

        # Update axes ranges and properties
        fig.update_yaxes(
            secondary_y=True,
            showgrid=True,
            gridcolor='#1f1f1f',
            gridwidth=1,
            fixedrange=False,
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1,
            spikecolor='#999999',
            uirevision=True,
            range=[price_min - price_margin * 0.1, price_max + price_margin]
        )

        fig.update_yaxes(
            secondary_y=False,
            showgrid=False,
            fixedrange=True,
            showticklabels=False,
            range=[0, volume_max * 2]
        )

        if timeframe != '1Y' and 'vwap' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['vwap'],
                    mode='lines',
                    line=dict(color='#bf40bf', width=2),
                    name='VWAP',
                    showlegend=False
                ),
                secondary_y=True
            )

            fig.add_annotation(
                text=f"VWAP: {df['vwap'].iloc[-1]:.2f}",
                xref='paper',
                x=1,
                yref='y2',
                y=df['vwap'].iloc[-1],
                xshift=40,
                yshift=20,  # Offset to not overlap with current pric
                showarrow=False,
                font=dict(size=11, color='#bf40bf'),
                bgcolor='black',
                bordercolor='#bf40bf',
                borderwidth=1,
                borderpad=4,
                opacity=1
            )

                                      
        return fig
        
    except Exception as e:
        print(f"Error creating analysis price chart: {str(e)}")
        traceback.print_exc()
        return create_empty_figure()

def calculate_vwap(df):
    """Calculate VWAP for the given dataframe"""
    df = df.copy()
    
    # Calculate price * volume
    df['pv'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    
    # Calculate cumulative values
    df['cumvol'] = df['volume'].cumsum()
    df['cumpv'] = df['pv'].cumsum()
    
    # Calculate VWAP
    df['vwap'] = df['cumpv'] / df['cumvol']
    
    return df['vwap']

def create_continuous_timeline(df):
    """Create continuous timeline by removing non-trading hours/days"""
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Convert to Eastern Time if needed
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('US/Eastern')
    elif df['time'].dt.tz != pytz.timezone('US/Eastern'):
        df['time'] = df['time'].dt.tz_convert('US/Eastern')
    
    # Filter for regular trading hours
    trading_mask = (
        ((df['time'].dt.hour == 9) & (df['time'].dt.minute >= 30)) |
        ((df['time'].dt.hour > 9) & (df['time'].dt.hour < 16))
    )
    df = df[trading_mask]
    
    # Sort by time
    df = df.sort_values('time')
    
    # Group by trading day
    df['date'] = df['time'].dt.date
    trading_days = df['date'].unique()
    
    # Create continuous timeline
    new_df = []
    base_time = df['time'].min()
    minutes_per_day = 390  # 6.5 hours * 60 minutes
    
    for i, day in enumerate(trading_days):
        day_data = df[df['date'] == day].copy()
        
        # Calculate minutes from market open
        market_open = pd.Timestamp(day).replace(hour=9, minute=30, tzinfo=pytz.timezone('US/Eastern'))
        minutes_from_open = (day_data['time'] - market_open).dt.total_seconds() / 60
        
        # Create new continuous timestamps
        day_data['time'] = base_time + pd.Timedelta(minutes=i * minutes_per_day) + pd.to_timedelta(minutes_from_open, unit='min')
        new_df.append(day_data)
    
    if new_df:
        df = pd.concat(new_df, ignore_index=True)
        df = df.drop('date', axis=1)
        
        # Convert back to Pacific Time for display
        df['time'] = df['time'].dt.tz_convert('US/Pacific')
    
    return df

def get_squeeze_status(df):
    """
    Get current squeeze status and count of bars in squeeze if active
    """
    if df is None or df.empty or 'Squeeze' not in df.columns:
        return "No data available", 0
    
    # Get current squeeze state
    current_squeeze = df['Squeeze'].iloc[-1]
    
    if current_squeeze:
        # Count consecutive squeeze bars from the end
        squeeze_bars = 0
        for i in range(len(df)-1, -1, -1):
            if df['Squeeze'].iloc[i]:
                squeeze_bars += 1
            else:
                break
        return "IN SQUEEZE", squeeze_bars
    else:
        return "NO SQUEEZE", 0

def get_market_holidays():
    """Return list of market holidays for the current year"""
    holidays = [
        '2024-01-01',  # New Year's Day
        '2024-01-15',  # Martin Luther King Jr. Day
        '2024-02-19',  # Presidents Day
        '2024-03-29',  # Good Friday
        '2024-05-27',  # Memorial Day
        '2024-06-19',  # Juneteenth
        '2024-07-04',  # Independence Day
        '2024-09-02',  # Labor Day
        '2024-11-28',  # Thanksgiving Day
        '2024-12-25'   # Christmas Day
    ]
    return holidays

def create_rsi_chart(df, timeframe):
    """Create RSI chart with continuous timeline"""
    if df is None or df.empty:
        return create_empty_figure()
        
    try:
        # Use continuous timeline for non-daily timeframes
        if timeframe != '1Y':
            df = create_continuous_timeline(df)
        
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['RSI'],
                name='RSI',
                line=dict(color='white', width=2),
                mode='lines'
            )
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought", annotation_position="left")
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold", annotation_position="left")
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            title="RSI Indicator",
            height=200,
            showlegend=False,
            plot_bgcolor='black',
            paper_bgcolor='black',
            xaxis=dict(
                showgrid=True,
                gridcolor='#1f1f1f',
                rangeslider=dict(visible=False),
                type="date",
                range=[df['time'].min(), df['time'].max() + pd.Timedelta(hours=1)],
                uirevision='same'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#1f1f1f',
                range=[0, 100],
                fixedrange=False  # Enable y-axis panning
            ),
            margin=dict(l=50, r=50, t=30, b=30),
            dragmode='pan'  # Enable panning by default
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating RSI chart: {str(e)}")
        traceback.print_exc()
        return create_empty_figure()

class SupplyDemandBin:
    def __init__(self, lvl, prev, sum_val=0, prev_sum=0, csum=0, avg=0, isreached=False):
        self.lvl = lvl
        self.prev = prev
        self.sum = sum_val
        self.prev_sum = prev_sum
        self.csum = csum
        self.avg = avg
        self.isreached = isreached

class VisibleRange:
    def __init__(self, threshold_percent=10, resolution=50):
        self.threshold_percent = threshold_percent
        self.resolution = resolution
        self.supply_box = None
        self.supply_avg = None
        self.supply_wavg = None
        self.demand_box = None
        self.demand_avg = None
        self.demand_wavg = None
        self.equilibrium = None
        self.weighted_equilibrium = None

    def calculate_zones(self, df):
        """Calculate supply and demand zones from OHLCV data"""
        if df is None or df.empty:
            return None

        try:
            max_price = df['high'].max()
            min_price = df['low'].min()
            total_volume = df['volume'].sum()
            csum = total_volume
            x1 = df.index[0]
            n = df.index[-1]

            range_size = (max_price - min_price) / self.resolution

            supply = SupplyDemandBin(max_price, max_price)
            demand = SupplyDemandBin(min_price, min_price)

            supply_found = False
            demand_found = False

            for i in range(self.resolution):
                supply.lvl -= range_size
                demand.lvl += range_size

                supply_mask = (df['high'] > supply.lvl) & (df['high'] < supply.prev)
                demand_mask = (df['low'] < demand.lvl) & (df['low'] > demand.prev)

                supply_volume = df.loc[supply_mask, 'volume'].sum()
                demand_volume = df.loc[demand_mask, 'volume'].sum()

                supply.sum += supply_volume
                supply.avg += supply.lvl * supply_volume
                supply.csum += supply_volume

                demand.sum += demand_volume
                demand.avg += demand.lvl * demand_volume
                demand.csum += demand_volume

                if (supply.sum / total_volume * 100 > self.threshold_percent) and not supply_found:
                    avg = (max_price + supply.lvl) / 2
                    supply_wavg = supply.avg / supply.csum if supply.csum > 0 else avg

                    self.supply_box = {
                        'top': max_price,
                        'bottom': supply.lvl,
                        'left': x1,
                        'right': n
                    }
                    self.supply_avg = avg
                    self.supply_wavg = supply_wavg
                    supply_found = True

                if (demand.sum / total_volume * 100 > self.threshold_percent) and not demand_found:
                    avg = (min_price + demand.lvl) / 2
                    demand_wavg = demand.avg / demand.csum if demand.csum > 0 else avg

                    self.demand_box = {
                        'top': demand.lvl,
                        'bottom': min_price,
                        'left': x1,
                        'right': n
                    }
                    self.demand_avg = avg
                    self.demand_wavg = demand_wavg
                    demand_found = True

                if supply_found and demand_found:
                    self.equilibrium = (max_price + min_price) / 2
                    self.weighted_equilibrium = (self.supply_wavg + self.demand_wavg) / 2
                    break

                supply.prev = supply.lvl
                demand.prev = demand.lvl

            return True

        except Exception as e:
            print(f"Error calculating supply/demand zones: {str(e)}")
            return None

    def add_zones_to_chart(self, fig, df):
        """Add supply and demand zones to the chart"""
        try:
            # Calculate extended time range (add 4 hours to the end)
            time_end = df['time'].max() + pd.Timedelta(hours=4)
            
            if self.supply_box:
                # Add supply zone shading between the lines
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), df['time'].min(), time_end, time_end],
                        y=[self.supply_box['bottom'], self.supply_box['top'], 
                           self.supply_box['top'], self.supply_box['bottom']],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                # Add supply zone lines
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.supply_box['bottom'], self.supply_box['bottom']],
                        mode='lines',
                        line=dict(color='red', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.supply_box['top'], self.supply_box['top']],
                        mode='lines',
                        line=dict(color='red', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                # Add supply average lines
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.supply_avg, self.supply_avg],
                        mode='lines',
                        line=dict(color='red', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.supply_wavg, self.supply_wavg],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )

            if self.demand_box:
                # Add demand zone shading between the lines
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), df['time'].min(), time_end, time_end],
                        y=[self.demand_box['bottom'], self.demand_box['top'], 
                           self.demand_box['top'], self.demand_box['bottom']],
                        fill="toself",
                        fillcolor="rgba(0, 255, 0, 0.1)",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                # Add demand zone lines
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.demand_box['bottom'], self.demand_box['bottom']],
                        mode='lines',
                        line=dict(color='green', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.demand_box['top'], self.demand_box['top']],
                        mode='lines',
                        line=dict(color='green', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                # Add demand average lines
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.demand_avg, self.demand_avg],
                        mode='lines',
                        line=dict(color='green', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['time'].min(), time_end],
                        y=[self.demand_wavg, self.demand_wavg],
                        mode='lines',
                        line=dict(color='green', width=1, dash='dash'),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    secondary_y=True
                )

        except Exception as e:
            print(f"Error adding zones to chart: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
