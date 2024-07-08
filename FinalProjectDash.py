#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:00:40 2024

@author: douglaseklund
"""

import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
from FinalProject import left_bank_stats
from FinalProject import stayed_bank_stats
from FinalProject import accuracy_rf
from FinalProject import report_df
import plotly.graph_objs as go

#%%

import plotly.io as pio
pio.renderers.default = 'browser'

#%%

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Credit Card Customer Statistics Dashboard'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': key, 'value': key} for key in left_bank_stats.keys()
        ],
        value='Card Category',
        clearable=False
    ),
    dcc.Graph(id='graph'),
    html.Div(children=[
        html.P('By comparing the different paramters the following where chosen in a RandomForestClassifier:'),
        html.Ul(children=[
            html.Li('Total Transaction Amount'),
            html.Li('Total Transaction Count'),
            html.Li('Change in Transaction Count (Q4 over Q1)'),
            html.Li('hange in Transaction Amount (Q4 over Q1)'),
            html.Li('Average Card Utilization Ratio'),
            html.Li('Total no. of products held by the customer'),
            html.Li('Total Revolving Balance on the Credit Card')
        ])
    ]),
    html.Div(children=f'The accuracy score of this model was {accuracy_rf.round(2)}% and the following table repsents the classification report:'),
    html.H1(''),
    dash_table.DataTable(
        id='classification-report',
        columns=[{'name': i, 'id': i} for i in report_df.columns],
        data=report_df.to_dict('records'),
        style_table={'width': '50%', 'margin': 'auto'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'center',
            'whiteSpace': 'normal',
            'height': 'auto',
        }) 
])

@app.callback(
    Output('graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_graph(selected_statistic):
    
    left_values = left_bank_stats[selected_statistic]
    stayed_values = stayed_bank_stats[selected_statistic]

    if isinstance(stayed_values, dict): 
        total_stayed = sum(stayed_values.values())
        total_left = sum(left_values.values())

        stayed_percentages = {}
        for key, value in stayed_values.items():
                stayed_percentages[key] = (value / total_stayed) * 100

        left_percentages = {}
        for key, value in left_values.items():
            left_percentages[key] = (value / total_left) * 100

        stayed_trace = go.Bar(x=list(stayed_percentages.keys()), y=list(stayed_percentages.values()), name='Stayed', marker=dict(color='rgb(28, 97, 42)'))
        left_trace = go.Bar(x=list(left_percentages.keys()), y=list(left_percentages.values()), name='Left', marker=dict(color='rgb(24, 66, 133)'))

        layout = go.Layout(barmode='group', 
                           title=f'Distribution of {selected_statistic} for Existing and Attrited Customers in %')
        fig = go.Figure(data=[stayed_trace, left_trace], layout=layout)
    
    else:
        stayed_trace = go.Table(
            header=dict(values=['Stayed', 'Left']),
            cells=dict(values=[[stayed_values], [left_values]])
        )

        layout = go.Layout(title=f'{selected_statistic} for Existing and Attrited Customers')
        fig = go.Figure(data=[stayed_trace], layout=layout)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug = True)    
    import socket
    host = socket.gethostbyname(socket.gethostname())
    port = 8050
    url = f'http://{host}:{port}'
    print(f'Dash application running on {url}/')


    