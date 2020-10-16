import os
import sys
import json
from textwrap import dedent

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from dash.dependencies import Input, Output, State
import fill_sunburst
from dash_sunburst import Sunburst
import dash_reusable_components as drc
from flask import Flask, render_template

RANDOM_STATE = 1130
array_of_MSE = []
array_of_score = []
array_of_features_used = []


layout = """

<!DOCTYPE html>
<html lang="en">
<title>{%title%}</title>
{%favicon%}
{%metas%}
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lato">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Montserrat">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
{%css%}
<style>
body,h1,h2,h3,h4,h5,h6 {font-family: "Lato", sans-serif}
.w3-bar,h1,button {font-family: "Montserrat", sans-serif}
.fa-anchor,.fa-coffee {font-size:200px}
</style>
<body>

<!-- First Grid -->
<div class="w3-row-padding w3-padding-64 w3-container">
  <div class="w3-content">
    <div class="w3-twothird">
      <h1>Boston Dataset</h1>
        {%app_entry%}
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
  <div class="d3js-visualization">
     <script>
     import React from "react";
import PropTypes from "prop-types"; 
import * as d3 from "d3";
import { withFauxDOM } from 'react-faux-dom'

class BarChartV4 extends React.Component {

    scaleColor = d3.scaleSequential(d3.interpolateViridis);
    scaleHeight = d3.scaleLinear();
    scaleWidth = d3.scaleBand().padding(0.1);

    componentDidMount() {
        this.updateChart();
    }

    componentDidUpdate (prevProps, prevState) { 
        if (this.props.data !== prevProps.data) {
            this.updateChart();
        }
    }

    updateChart() {
    this.updateScales();

    const { data, width, height, animDuration } = this.props;
        const faux = this.props.connectFauxDOM("g", "chart");
        const bars = d3.select(faux)
                            .selectAll(".bar")
                            .data(data, function key(d) { return d.item });
        bars.exit()
            .transition().duration(animDuration)
                .attr("y", height)
                .attr("height", 0)
                .style("fill-opacity", 0)
            .remove();

        bars.enter()
            .append("rect")
                .attr("class", "bar")
                .attr("y", height)
                .attr("x", width )
                .attr("width", 0)
                .attr("height", 0)
                .attr("rx", 5 ).attr("ry", 5 )
            .merge(bars)
                .transition().duration(animDuration)
                .attr("y", (d) => ( this.scaleHeight(d.count) ))
                .attr("height", (d) => (height - this.scaleHeight(d.count)) )
                .attr("x", (d, i) => ( this.scaleWidth(d.item) ) )
                .attr("width", this.scaleWidth.bandwidth() )
                .style("fill",  (d, i) => ( this.scaleColor(i) ));

        this.props.animateFauxDOM(800);
    }

    updateScales() {
        const { data, width, height } = this.props;
        this.scaleColor.domain([0, data.length]);
        this.scaleWidth
                 .domain(data.map((d) => (d.item)))
                 .range([0, width]);
        this.scaleHeight
                 .domain(d3.extent(data, (d) => (d.count)))
                 .range([height - 20, 0]);
    }

    render() {
        const { width, height } = this.props;
        return (
            <svg width={width} height={height} >
                { this.props.chart }
            </svg>
        );    
    }
}

BarChartV4.defaultProps = {
    animDuration: 600
};

BarChartV4.propTypes = {
     data: PropTypes.array.isRequired,
      width: PropTypes.number.isRequired,
     height: PropTypes.number.isRequired,
     animDuration: PropTypes.number
};

export default withFauxDOM(BarChartV4);
    });
}
 $(document).click(function(evt) {
      outputStatement();
    });
    $("#catDIV, #mouseDIV").click(function(evt) {
      evt.stopPropagation();
    });
</script>
</div>
  {%config%}
  {%scripts%}
  {%renderer%}
</body>
</html>
"""

sunburst_data_scores = {
    'name': 'boston',
    'children': [
        {
            'name': 'Lstat',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='LSTAT', error_bool=True, score_bool=False
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='LSTAT', error_bool=True, score_bool=False
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='LSTAT', error_bool=True, score_bool=False
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='LSTAT', error_bool=True, score_bool=False
                )}
            ]
        },
        {
            'name': 'Indus',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='INDUS', error_bool=True, score_bool=False
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='INDUS', error_bool=True, score_bool=False
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='INDUS', error_bool=True, score_bool=False
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='INDUS', error_bool=True, score_bool=False
                )}
            ]
        },
        {
            'name': 'Nox',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='NOX', error_bool=True, score_bool=False
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='NOX', error_bool=True, score_bool=False
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='NOX', error_bool=True, score_bool=False
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='NOX', error_bool=True, score_bool=False
                )}
            ]
        },
{
            'name': 'Rm',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='RM', error_bool=True, score_bool=False
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='RM', error_bool=True, score_bool=False
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='RM', error_bool=True, score_bool=False
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='RM', error_bool=True, score_bool=False
                )}
            ]
        },
{
            'name': 'Age',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='AGE', error_bool=True, score_bool=False
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='AGE', error_bool=True, score_bool=False
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='AGE', error_bool=True, score_bool=False
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='AGE', error_bool=True, score_bool=False
                )}
            ]
        },
{
            'name': 'Tax',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='TAX', error_bool=True, score_bool=False
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='TAX', error_bool=True, score_bool=False
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='TAX', error_bool=True, score_bool=False
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='TAX', error_bool=True, score_bool=False
                )}
            ]
        },
    ]

}
sunburst_data_errors = {
    'name': 'boston',
    'children': [
        {
            'name': 'Lstat',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='LSTAT', error_bool=False, score_bool=True
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='LSTAT', error_bool=False, score_bool=True
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='LSTAT', error_bool=False, score_bool=True
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='LSTAT', error_bool=False, score_bool=True
                )}
            ]
        },
        {
            'name': 'Indus',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='INDUS', error_bool=False, score_bool=True
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='INDUS', error_bool=False, score_bool=True
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='INDUS', error_bool=False, score_bool=True
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='INDUS', error_bool=False, score_bool=True
                )}
            ]
        },
        {
            'name': 'Nox',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='NOX', error_bool=False, score_bool=True
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='NOX', error_bool=False, score_bool=True
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='NOX', error_bool=False, score_bool=True
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='NOX', error_bool=False, score_bool=True
                )}
            ]
        },
{
            'name': 'Rm',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='RM', error_bool=False, score_bool=True
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='RM', error_bool=False, score_bool=True
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='RM', error_bool=False, score_bool=True
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='RM', error_bool=False, score_bool=True
                )}
            ]
        },
{
            'name': 'Age',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='AGE', error_bool=False, score_bool=True
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='AGE', error_bool=False, score_bool=True
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='AGE', error_bool=False, score_bool=True
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='AGE', error_bool=False, score_bool=True
                )}
            ]
        },
{
            'name': 'Tax',
            'children': [
                {'name': 'Linear', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='linear', feature_names='TAX', error_bool=False, score_bool=True
                )},
                {'name': 'Lasso', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='lasso', feature_names='TAX', error_bool=False, score_bool=True
                )},
                {'name': 'Ridge', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='ridge', feature_names='TAX', error_bool=False, score_bool=True
                )},
                {'name': 'Elastic', 'size': fill_sunburst.fill_sunburst_scores_errors(
                    regression_model_name='elastic_net', feature_names='TAX', error_bool=False, score_bool=True
                )}
            ]
        },
    ]

}
app = dash.Dash(__name__, serve_locally=False, index_string=layout)
server = app.server
app.layout = html.Div([
    html.Div(className="banner", children=[
        html.Div(className='container scalable', style={"text-align": "center"}, children=[
            html.H1(style={'backgroundImage': 'https://krot.info/uploads/posts/2020-01/1579386182_24-64.jpg'}),
            html.H2(html.A(
                'Regression Feature Selection',
                style={'textAlign': 'center', 'color': 'blue'}
            )),
            html.A(
                html.Img(src="https://www.wemakescholars.com/admin/uploads/providers/35reiq6enBmZkohTfbNPlCKM758T7wdP.jpg"),
                href='https://www.hs-aalen.de/',
                style={'backgroundColor': 'blue'}
            )
        ]),
    ]),

    html.Div(id='body', className='container scalable', style={"background-color":"#f2f5f5"}, children=[
        html.Div(
            className='row',
            style={'paddingBottom': '10px'},
            children=dcc.Markdown(dedent("""
            Nothing feels like 127.0.0.1
            """))
        ),

        html.Div(id='custom-data-storage', style={'display': 'none'}),

        html.Div(className='row', children=[
            html.Div(className='four columns', children=drc.NamedDropdown(
                name='Select Feature',
                id='dropdown-feature',
                options=[
                    {'label': 'INDUS', 'value': 'INDUS'},
                    {'label': 'LSTAT', 'value': 'LSTAT'},
                    {'label': 'Custom Data', 'value': 'custom'},
                    {'label': 'NOX', 'value': 'NOX'},
                    {'label': 'RM', 'value': 'RM'},
                    {'label': 'AGE', 'value': 'AGE'},
                    {'label': 'TAX', 'value': 'TAX'},
                ],
                value='LSTAT',
                clearable=False,
                searchable=False,
            )),

            html.Div(className='four columns', children=drc.NamedDropdown(
                name='Select Model',
                id='dropdown-select-model',
                options=[
                    {'label': 'Linear Regression', 'value': 'linear'},
                    {'label': 'Lasso', 'value': 'lasso'},
                    {'label': 'Ridge', 'value': 'ridge'},
                    {'label': 'Elastic Net', 'value': 'elastic_net'},
                ],
                value='linear',
                searchable=False,
                clearable=False
            )),

            html.Div(className='four columns', children=drc.NamedDropdown(
                name='Click Mode (Select Custom Data to enable)',
                id='dropdown-custom-selection',
                options=[
                    {'label': 'Add Training Data', 'value': 'training'},
                    {'label': 'Add Test Data', 'value': 'test'},
                    {'label': 'Remove Data point', 'value': 'remove'},
                    {'label': 'Do Nothing', 'value': 'nothing'},
                ],
                value='training',
                clearable=False,
                searchable=False
            )),
        ]),

        html.Div(className='row', children=[
            html.Div(className='four columns', children=drc.NamedSlider(
                name='Polynomial Degree',
                id='slider-polynomial-degree',
                min=1,
                max=10,
                step=1,
                value=1
            )),

            html.Div(className='four columns', children=drc.NamedSlider(
                name='Alpha (Regularization Term)',
                id='slider-alpha',
                min=-4,
                max=3,
                value=0,
                marks={i: '{}'.format(10 ** i) for i in range(-4, 4)}
            )),

            html.Div(
                className='four columns',
                style={
                    'overflowX': 'hidden',
                    'overflowY': 'visible',
                    'paddingBottom': '10px'
                },
                children=drc.NamedSlider(
                    name='L1/L2 ratio (Select Elastic Net to enable)',
                    id='slider-l1-l2-ratio',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5,
                    marks={0: 'L1', 1: 'L2'}
                )
            ),
        ]),

        dcc.Graph(
            id='graph-regression-display',
            className='row',
            style={'height': 'calc(100vh - 160px)'},
            config={'modeBarButtonsToRemove': [
                'pan2d',
                'lasso2d',
                'select2d',
                'autoScale2d',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'toggleSpikelines'
            ]}
        ),
    ]),
    html.Div(className='d3_vis', style={'text-align':'center'}, children=[
            html.H3(html.A(
                'D3 sunburst visualization ',
                style={'text-align': 'center', 'color': 'blue'}
            )),
            html.H4(html.A(
                'Score/Error'
            ))
    ]),
    html.Div(className='d3', children=[
            html.Div([
                html.Div(
                    [Sunburst(id='sun', data=sunburst_data_scores)],
                    style={'width': '49%', 'display': 'inline-block', 'float': 'left'})
            ])
    ]),
    html.Div(className='d3_2', children=[
        html.Div(
           [Sunburst(id='sun_2', data=sunburst_data_errors)],
           style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
        ]),
    ])



def make_dataset(name, random_state):
    np.random.seed(random_state)
    if name != None:
        array_of_features_used.append(name)
    if name == "INDUS":
        X = load_boston().data[:, 2].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'NOX':
        X = load_boston().data[:, 4].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'LSTAT':
        X = load_boston().data[:, -1].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'RM':
        X = load_boston().data[:, 5].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'AGE':
        X = load_boston().data[:, 6].reshape(-1, 1)
        y = load_boston().target
        return X, y
    elif name == 'TAX':
        X = load_boston().data[:, -4].reshape(-1, 1)
        y = load_boston().target
        return X, y

    else:
        return make_regression(n_samples=300, n_features=1, noise=20,
                               random_state=random_state)

# print(make_dataset(name='boston', random_state=100))
def format_coefs(coefs):
    coef_string = "yhat = "
    for order, coef in enumerate(coefs):
        if coef >= 0:
            sign = ' + '
        else:
            sign = ' - '
        if order == 0:
            coef_string += f'{coef}'
        elif order == 1:
            coef_string += sign + f'{abs(coef):.3f}*x'
        else:
            coef_string += sign + f'{abs(coef):.3f}*x^{order}'

    return coef_string

    def fill_sunburst_scores_errors(regression_model_name, feature_names, error_bool, score_bool):

        x1, y1 = make_dataset(dataset=feature_names, random_state=RANDOM_STATE)
        if regression_model_name == 'lasso':
            models = Lasso(normalize=True)
        elif regression_model_name == 'ridge':
            models = Ridge(normalize=True)
        elif regression_model_name == 'elastic_net':
            models = ElasticNet(normalize=True)
        else:
            models = LinearRegression(normalize=True)
        x_trains, x_tests, y_trains, y_tests = \
            train_test_split(x1, y1, test_size=100, random_state=RANDOM_STATE)
        test_scores = models.score(x_trains, y_trains)
        test_errors = mean_squared_error(y_tests, models.predict(x_tests))
        if score_bool == True:
            return test_scores
        elif error_bool == True:
            return test_errors

    print(fill_sunburst_scores_errors(regression_model_name='ridge', feature_names='LSTAT', error_bool=True, score_bool=False))
@app.callback(Output('slider-alpha', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_slider_alpha(model):
    return model not in ['lasso', 'ridge', 'elastic_net']


@app.callback(Output('slider-l1-l2-ratio', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_dropdown_select_model(model):
    return model not in ['elastic_net']


@app.callback(Output('dropdown-custom-selection', 'disabled'),
              [Input('dropdown-feature', 'value')])
def disable_custom_selection(dataset):
    return dataset != 'custom'

@app.callback(Output('custom-data-storage', 'children'),
              [Input('graph-regression-display', 'clickData')],
              [State('dropdown-custom-selection', 'value'),
               State('custom-data-storage', 'children'),
               State('dropdown-feature', 'value')])
def update_custom_storage(clickData, selection, data, dataset):
    if data is None:
        data = {
            'train_X': [1, 2],
            'train_y': [1, 2],
            'test_X': [3, 4],
            'test_y': [3, 4],
        }
    else:
        data = json.loads(data)
        if clickData and dataset == 'custom':
            selected_X = clickData['points'][0]['x']
            selected_y = clickData['points'][0]['y']

            if selection == 'training':
                data['train_X'].append(selected_X)
                data['train_y'].append(selected_y)
            elif selection == 'test':
                data['test_X'].append(selected_X)
                data['test_y'].append(selected_y)
            elif selection == 'remove':
                while selected_X in data['train_X'] and selected_y in data['train_y']:
                    data['train_X'].remove(selected_X)
                    data['train_y'].remove(selected_y)
                while selected_X in data['test_X'] and selected_y in data['test_y']:
                    data['test_X'].remove(selected_X)
                    data['test_y'].remove(selected_y)

    return json.dumps(data)


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('dropdown-feature', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-alpha', 'value'),
               Input('dropdown-select-model', 'value'),
               Input('slider-l1-l2-ratio', 'value'),
               Input('custom-data-storage', 'children')])
def update_graph(dataset, degree, alpha_power, model_name, l2_ratio, custom_data):
    # Generate base data
    if dataset == 'custom':
        custom_data = json.loads(custom_data)
        X_train = np.array(custom_data['train_X']).reshape(-1, 1)
        y_train = np.array(custom_data['train_y'])
        X_test = np.array(custom_data['test_X']).reshape(-1, 1)
        y_test = np.array(custom_data['test_y'])
        X_range = np.linspace(-5, 5, 300).reshape(-1, 1)
        X = np.concatenate((X_train, X_test))

        trace_contour = go.Contour(
            x=np.linspace(-5, 5, 300),
            y=np.linspace(-5, 5, 300),
            z=np.ones(shape=(300, 300)),
            showscale=False,
            hoverinfo='none',
            contours=dict(coloring='lines'),
        )
    else:
        X, y = make_dataset(dataset, RANDOM_STATE)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=100, random_state=RANDOM_STATE)

        X_range = np.linspace(X.min() - 0.5, X.max() + 0.5, 300).reshape(-1, 1)

    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # Create Polynomial Features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    poly_range = poly.fit_transform(X_range)

    # Select model
    alpha = 10 ** alpha_power
    # print(alpha)
    if model_name == 'lasso':
        model = Lasso(alpha=alpha, normalize=True)
    elif model_name == 'ridge':
        model = Ridge(alpha=alpha, normalize=True)
    elif model_name == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=1 - l2_ratio, normalize=True)
    else:
        model = LinearRegression(normalize=True)

    # Train model and predict
    # print(model_name)
    # print(model)
    model.fit(X_train_poly, y_train)
    y_pred_range = model.predict(poly_range)
    test_score = model.score(X_test_poly, y_test)
    test_error = mean_squared_error(y_test, model.predict(X_test_poly))
    # print(test_error)
    # print(X_train_poly)

    array_of_MSE.append(test_error)
    array_of_score.append(test_score)
    array_of_mse = list(set(array_of_MSE))
    array_of_scores = list(set(array_of_score))
    array_of_features = list(set(array_of_features_used))
    # print(array_of_scores, array_of_mse, array_of_features)
    # Create figure
    trace0 = go.Scatter(
        x=X_train.squeeze(),
        y=y_train,
        name='Training Data',
        mode='markers',
        opacity=0.7,
    )
    trace1 = go.Scatter(
        x=X_test.squeeze(),
        y=y_test,
        name='Test Data',
        mode='markers',
        opacity=0.7,
    )
    trace2 = go.Scatter(
        x=X_range.squeeze(),
        y=y_pred_range,
        name='Prediction',
        mode='lines',
        hovertext=format_coefs(model.coef_)
    )
    data = [trace0, trace1, trace2]
    if dataset == 'custom':
        data.insert(0, trace_contour)

    layout = go.Layout(
        title=f"Score: {test_score:.3f}, MSE: {test_error:.3f} (Test Data)",
        legend=dict(orientation='h'),
        margin=dict(l=25, r=25),
        hovermode='closest'
    )

    return go.Figure(data=data, layout=layout)


external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})
# Running the server
html.Script(src="C://Users/orbitronic/PycharmProjects/untitled2/myjs.js")

if __name__ == '__main__':
    app.run_server(debug=True)
