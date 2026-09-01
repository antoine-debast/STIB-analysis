# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 18:04:34 2025

@author: Aurélien
"""

# Import packages
from dash import Dash, html, dcc, callback, Output, Input
from datetime import date
#import dash_ag_grid as dag
import pandas as pd
import geopandas as gpd
import geopandas.accessors
import plotly.express as px
data_in = "data/fusion result/"

# Date range covering the project scope
sdate_debut = "2025-08-15"
sdate_fin = "2025-09-15"
sdate=sdate_debut

# Incorporate data
data_in = "data/punctuality_results/punctuality_results.pkl" 
punc = pd.read_pickle(data_in)
punc = punc.reset_index()
punc=punc.set_geometry('stop_geo')
punc['mean_delay_min']=punc['mean_delay'].dt.total_seconds()/60
punc["date_string"] = pd.to_datetime(punc['date'], infer_datetime_format=True).dt.strftime('%Y-%m-%d')
punc["Pos_x"]=punc['stop_geo'].geo.x
punc["Pos_y"]=punc['stop_geo'].geo.y
# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children='Mean delay by hour'),
    html.Div([
    dcc.RadioItems(['hourly', 'all'], 'hourly', id='my-final-radio-item-example'),
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=date(2025, 8, 15),
        max_date_allowed=date(2025, 9, 15),
        initial_visible_month=date(2025, 9, 2),
        date=date(2025, 9, 2)
    ),
    dcc.Dropdown(
                punc['route_short_name'].unique(),
                '71',
                id='crossfilter-xaxis-column',
    )]),
    html.Hr(),
    
    dcc.RadioItems(options=[
       {'label': 'Aller', 'value': True},
       {'label': 'Retour', 'value': False},
   ],
   value=True, id='radio-direction', inline=True),
    # dag.AgGrid(
    #     rowData=punc.to_dict('records'),
    #     columnDefs=columnDefs,
    #     defaultColDef={"flex": 1, "minWidth": 150, "floatingFilter": True},
    #         dashGridOptions={"animateRows": False}
    # ),
    dcc.Graph(figure={}, id='my-final-graph-example'),
    dcc.Graph(
        id='stop_positions',
        #figure=px.scatter_map(punc, lat="Pos_x", lon="Pos_y", color="stop_name")
        figure=px.scatter(punc, x="Pos_x", y="Pos_y", color="stop_name")
    )
]

# Add controls to build the interaction
@callback(
    Output(component_id='my-final-graph-example', component_property='figure'),
    Output(component_id='stop_positions', component_property='figure'),
    Input(component_id='my-final-radio-item-example', component_property='value'),
    Input(component_id='radio-direction', component_property='value'),
    Input(component_id='my-date-picker-single', component_property='date'),
    Input('crossfilter-xaxis-column', 'value'),
)
def update_graph(radio_set,dir,date_value,xaxis_column_name):
    global punc
    date_object = date.fromisoformat(date_value)
    if((radio_set=='all')):
        punc_fil=punc.query('(date==@date_object)')
        if(not ('timestamp_dt_mean' in punc)or(len(punc_fil)==0)):
            data_in = f"data/fusion result/Fused_GTFS_VehiclePositions({date_value}).pkl" 
            punc = pd.read_pickle(data_in)
            punc['delay_min']=punc['delay'].dt.total_seconds()/60
            punc['date']=punc['timestamp_dt_mean'].dt.date
            punc["date_string"] = pd.to_datetime(punc['date'], infer_datetime_format=True).dt.strftime('%Y-%m-%d')
            punc["Pos_x"]=punc['stop_geo'].geo.x
            punc["Pos_y"]=punc['stop_geo'].geo.y
        
    else:
        if(not ('mean_delay' in punc)):
            data_in = "data/punctuality_results/punctuality_results.pkl" 
            punc = pd.read_pickle(data_in)
            punc = punc.reset_index()
            punc['mean_delay_min']=punc['mean_delay'].dt.total_seconds()/60
            punc["date_string"] = pd.to_datetime(punc['date'], infer_datetime_format=True).dt.strftime('%Y-%m-%d')
            punc["Pos_x"]=punc['stop_geo'].geo.x
            punc["Pos_y"]=punc['stop_geo'].geo.y
        
    
    if('timestamp_dt_mean' in punc):
        punc_fil=punc.query('(date==@date_object)&(route_short_name==@xaxis_column_name)&(direction_id==@dir)')
        fig = px.scatter(punc_fil, x='timestamp_dt_mean', y='delay_min', color='stop_name')
        #map=px.scatter_map(punc_fil, lon="Pos_x", lat="Pos_y", color="stop_name")
        map=px.scatter(punc_fil, x="Pos_x", y="Pos_y", color="stop_name")
    else:
        punc_fil=punc.query('(date==@date_object)&(route_short_name==@xaxis_column_name)&(direction_id==@dir)')
        fig = px.scatter(punc_fil, x='actual_hour', y='mean_delay_min', color='stop_name')
        #map=px.scatter_map(punc_fil, lon="Pos_x", lat="Pos_y", color="stop_name")
        map=px.scatter(punc_fil, x="Pos_x", y="Pos_y", color="stop_name")
    return fig, map

# Run the app
if __name__ == '__main__':
    app.run(debug=True)