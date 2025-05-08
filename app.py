#!/usr/bin/env python
# coding: utf-8

# In[13]:


import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output
import numpy as np


# In[44]:


#LOAD THE DATA
data = pd.read_csv("sample.csv").drop(columns=['Unnamed: 0'])

# Define categorical columns for dropdown
categorical_cols = data.select_dtypes(include='object').columns
numerical_cols = data.select_dtypes(include=['int', 'float']).columns.drop(['id', 'click'])


# In[45]:


# LOAD THE LOGISTIC MODEL DATA
log_results_df = pd.read_csv("log_results_df.csv")
log_feature_importance = pd.read_csv("log_feature_importance.csv")
log_y_test = log_results_df['y_test']
log_test_pred_proba = log_results_df['test_pred_proba']
log_test_pred = log_results_df['test_pred']

log_fpr, log_tpr, log_thresholds = roc_curve(log_y_test, log_test_pred_proba)
auc_score = roc_auc_score(log_y_test, log_test_pred_proba)
log_feature_importance_sorted = log_feature_importance.sort_values(by='Importance')

# Build the Plotly Figures
log_roc_fig = go.Figure()
log_roc_fig.add_trace(go.Scatter(x=log_fpr, y=log_tpr, mode='lines', name=f'ROC curve (AUC = {auc_score:.4f})', line=dict(color='orange')))
log_roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier', line=dict(color='navy', dash='dash')))
log_roc_fig.update_layout(title='ROC Curve of Logistic Regression', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

log_cm = confusion_matrix(log_y_test, log_test_pred)
labels = np.array([['TN', 'FP'], ['FN', 'TP']])

log_cm_fig = go.Figure(data=go.Heatmap(z=np.flipud(log_cm), x=['Pred 0', 'Pred 1'], y=['True 1', 'True 0'], colorscale='Blues', text=np.flipud(log_cm), texttemplate="%{text}",
                                       hoverinfo='text'))
log_cm_fig.update_layout(title='Confusion Matrix (Logistic Regression)', xaxis_title='Predicted', yaxis_title='Actual')

log_importance_fig = px.bar(log_feature_importance_sorted, x='Importance', y='Feature', orientation='h', title='Logistic Regression Feature Importance', 
                            labels={'Importance': 'Importance Score', 'Feature': 'Feature'}, height=700)


# In[46]:


# BASELINE MODEL
baseline_y_test = log_y_test  # reuse test set
baseline_test_pred = np.zeros_like(baseline_y_test)  # predict all 0s
baseline_test_pred_proba = np.full_like(baseline_y_test, fill_value=0.0, dtype=float)

baseline_fpr, baseline_tpr, _ = roc_curve(baseline_y_test, baseline_test_pred_proba)
baseline_auc_score = roc_auc_score(baseline_y_test, baseline_test_pred_proba)

baseline_cm = confusion_matrix(baseline_y_test, baseline_test_pred)

# Build the Plotly Figures
baseline_roc_fig = go.Figure()
baseline_roc_fig.add_trace(go.Scatter(x=baseline_fpr, y=baseline_tpr, mode='lines', name=f'ROC curve (AUC = {baseline_auc_score:.4f})', 
                                      line=dict(color='orange')))
baseline_roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier', line=dict(color='navy', dash='dash')))
baseline_roc_fig.update_layout(title='ROC Curve of Baseline Model', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

baseline_cm_fig = go.Figure(data=go.Heatmap(z=np.flipud(baseline_cm), x=['Pred 0', 'Pred 1'], y=['True 1', 'True 0'], colorscale='Blues',
    text=np.flipud(baseline_cm), texttemplate="%{text}", hoverinfo='text'))
baseline_cm_fig.update_layout(title='Confusion Matrix (Baseline Model)', xaxis_title='Predicted', yaxis_title='Actual')


# In[47]:


# LOAD THE NEURAL NETWORK DATA
mlp_results_df = pd.read_csv("mlp_results_df.csv")
mlp_y_test = mlp_results_df['y_test']
mlp_test_pred_proba = mlp_results_df['test_pred_proba']
mlp_test_pred = mlp_results_df['test_pred']

mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(mlp_y_test, mlp_test_pred_proba)
auc_score = roc_auc_score(mlp_y_test, mlp_test_pred_proba)

# Build the Plotly Figures
mlp_roc_fig = go.Figure()
mlp_roc_fig.add_trace(go.Scatter(x=mlp_fpr, y=mlp_tpr, mode='lines', name=f'ROC curve (AUC = {auc_score:.4f})', line=dict(color='orange')))
mlp_roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier', line=dict(color='navy', dash='dash')))
mlp_roc_fig.update_layout(title='ROC Curve of Neural Network (MLP)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

mlp_cm = confusion_matrix(mlp_y_test, mlp_test_pred)
labels = np.array([['TN', 'FP'], ['FN', 'TP']])

mlp_cm_fig = go.Figure(data=go.Heatmap(z=np.flipud(mlp_cm), x=['Pred 0', 'Pred 1'], y=['True 1', 'True 0'], colorscale='Blues', text=np.flipud(mlp_cm), 
                                       texttemplate="%{text}", hoverinfo='text'))
mlp_cm_fig.update_layout(title='Confusion Matrix (Neural Network (MLP))', xaxis_title='Predicted', yaxis_title='Actual')


# In[48]:


# LOAD THE RANDOM FOREST MODEL DATA
rf_results_df = pd.read_csv("rf_results_df.csv")
rf_feature_importance = pd.read_csv("rf_feature_importance.csv")
rf_y_test = rf_results_df['y_test']
rf_test_pred_proba = rf_results_df['test_pred_proba']
rf_test_pred = rf_results_df['test_pred']

rf_fpr, rf_tpr, rf_thresholds = roc_curve(rf_y_test, rf_test_pred_proba)
rf_auc_score = roc_auc_score(rf_y_test, rf_test_pred_proba)
rf_feature_importance_sorted = rf_feature_importance.sort_values(by='Importance')

# Build the Plotly Figures
rf_roc_fig = go.Figure()
rf_roc_fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines', name=f'ROC curve (AUC = {rf_auc_score:.4f})', line=dict(color='orange')))
rf_roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier', line=dict(color='navy', dash='dash')))
rf_roc_fig.update_layout(title='ROC Curve of Random Forest', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

rf_cm = confusion_matrix(rf_y_test, rf_test_pred)
labels = np.array([['TN', 'FP'], ['FN', 'TP']])

rf_cm_fig = go.Figure(data=go.Heatmap(z=np.flipud(rf_cm), x=['Pred 0', 'Pred 1'], y=['True 1', 'True 0'], colorscale='Blues', text=np.flipud(rf_cm), texttemplate="%{text}",
                                       hoverinfo='text'))
rf_cm_fig.update_layout(title='Confusion Matrix (Random forest)', xaxis_title='Predicted', yaxis_title='Actual')

rf_importance_fig = px.bar(rf_feature_importance_sorted, x='Importance', y='Feature', orientation='h', title='Random Forest Feature Importance', 
                            labels={'Importance': 'Importance Score', 'Feature': 'Feature'}, height=700)


# In[49]:


# Initialize app
app = JupyterDash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("CTR Project Dashboard"),
    dcc.Tabs(id='tabs', value='tab-preprocess', children=[
        dcc.Tab(label='Data Analysis before Modeling', value='tab-preprocess'),
        dcc.Tab(label='Model Analysis', value='tab-model')
    ]),
    html.Div(id='tabs-content')
])

# Define callback for tab content
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-preprocess':
        return html.Div([
            html.H3("CTR by each Category"),
            dcc.Dropdown(
                id='eda-feature-dropdown',
                options=[{'label': col, 'value': col} for col in categorical_cols],
                value='app_id',
                clearable=False
            ),
            dcc.Slider(
                id='top-n-slider',
                min=5, max=20, step=5, value=10,
                marks={5: 'Top 5', 10: 'Top 10', 15: 'Top 15', 20: 'Top 20'}
            ),
            dcc.RadioItems(
                id='chart-type-radio',
                options=[{'label': 'Count', 'value': 'count'},
                         {'label': 'CTR', 'value': 'ctr'}],
                value='count'
            ),
            dcc.Graph(id='eda-graph'),

            html.H3("Correlation Heatmap"),
            dcc.Checklist(
                id='corr-feature-checklist',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[:3],  # default to first 3
                inline=True
            ),
            dcc.Graph(id='corr-graph'),

            html.H3("CTR by Time Filter"),
            dcc.RangeSlider(
                id='time-range-slider',
                min=0, max=23, step=1, value=[0, 23],
                marks={i: str(i) for i in range(0, 24)}
            ),
            dcc.Dropdown(
                id='time-feature-dropdown',
                options=[
                    {'label': 'Hour of Day', 'value': 'hour_of_day'},
                    {'label': 'Day of Week (0 is Monday)', 'value': 'day_of_week'}
                ],
                value='hour_of_day',
                clearable=False
            ),
            dcc.Graph(id='ctr-time-graph')
        ])
        
    elif tab == 'tab-model':
        return html.Div([
            html.H2("Model Analysis"),
            dcc.Dropdown(
                id='model-selection-dropdown',
                options=[
                    {'label': 'Baseline', 'value': 'baseline'},
                    {'label': 'Logistic Regression', 'value': 'logistic'},
                    {'label': 'Neural Network', 'value': 'neuralnetwork'},
                    {'label': 'Random Forest', 'value': 'randomforest'}
                ],
                value='logistic',
                clearable=False
            ),
            dcc.RadioItems(
                id='metric-radio',
                options=[
                    {'label': 'AUC', 'value': 'auc'},
                    {'label': 'Confusion Matrix', 'value': 'confusion'},
                    {'label': 'Feature Importance (N/A for baseline and MLP)', 'value': 'importance'}
                ],
                value='auc'
            ),
            dcc.Graph(id='model-graph')
        ])

# Callback for EDA plot
@app.callback(
    Output('eda-graph', 'figure'),
    Input('eda-feature-dropdown', 'value'),
    Input('top-n-slider', 'value'),
    Input('chart-type-radio', 'value')
)

def update_eda_plot(feature, top_n, plot_type):
    top_categories = data[feature].value_counts().nlargest(top_n).index
    filtered_data = data[data[feature].isin(top_categories)]

    if plot_type == 'count':
        counts = filtered_data[feature].value_counts()
        fig = px.bar(x=counts.index, y=counts.values,
                     labels={'x': feature, 'y': 'Count'},
                     title=f"Top {top_n} {feature} Categories by Count")
    elif plot_type == 'ctr':
        ctr = filtered_data.groupby(feature)['click'].mean().sort_values(ascending=False)
        ctr = ctr.loc[top_categories]  # ensure same top order
        fig = px.bar(x=ctr.index, y=ctr.values,
                     labels={'x': feature, 'y': 'CTR'},
                     title=f"Top {top_n} {feature} Categories by CTR")
    return fig

# Correlation heatmap
@app.callback(
    Output('corr-graph', 'figure'),
    Input('corr-feature-checklist', 'value')
)
def update_corr_plot(selected_features):
    corr_matrix = data[selected_features].corr()
    fig = px.imshow(corr_matrix, text_auto=True,
                    title="Correlation Heatmap")
    return fig

# CTR over time by hour filter
@app.callback(
    Output('ctr-time-graph', 'figure'),
    Input('time-feature-dropdown', 'value'),
    Input('time-range-slider', 'value')
)
def update_ctr_time(time_feature, selected_range):
    filtered = data[(data[time_feature] >= selected_range[0]) & (data[time_feature] <= selected_range[1])]
    ctr = filtered.groupby(time_feature)['click'].mean().sort_index()
    fig = px.line(x=ctr.index, y=ctr.values,
                  labels={'x': time_feature.replace('_', ' ').title(), 'y': 'CTR'},
                  title=f"CTR by {time_feature.replace('_', ' ').title()} (Range {selected_range[0]}–{selected_range[1]})")
    return fig

# Update Slider for CTR by time
@app.callback(
    Output('time-range-slider', 'min'),
    Output('time-range-slider', 'max'),
    Output('time-range-slider', 'marks'),
    Output('time-range-slider', 'value'),
    Input('time-feature-dropdown', 'value')
)
def update_slider_bounds(time_feature):
    if time_feature == 'hour_of_day':
        min_val, max_val = 0, 23
        marks = {i: str(i) for i in range(0, 24)}
        value = [0, 23]
    else:  # day_of_week
        min_val, max_val = 0, 6
        marks = {i: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i] for i in range(0, 7)}
        value = [0, 6]
    return min_val, max_val, marks, value

# Callback for Model Analysis plot
@app.callback(
    Output('model-graph', 'figure'),
    Input('model-selection-dropdown', 'value'),
    Input('metric-radio', 'value')
)
def update_model_plot(model, metrics):
    if model == 'logistic':
        if metrics == 'auc':
            return log_roc_fig
        elif metrics == 'confusion':
            return log_cm_fig
        elif metrics == 'importance':
            return log_importance_fig 
    elif model == 'neuralnetwork':
        if metrics == 'auc':
            return mlp_roc_fig
        elif metrics == 'confusion':
            return mlp_cm_fig
        else:
            return go.Figure()
    elif model == 'baseline':
        if metrics == 'auc':
            return baseline_roc_fig
        elif metrics == 'confusion':
            return baseline_cm_fig
        else:
            return go.Figure()
    elif model == 'randomforest':
        if metrics == 'auc':
            return rf_roc_fig
        elif metrics == 'confusion':
            return rf_cm_fig
        else:
            return rf_importance_fig
    else:
        return go.Figure()

# Run app
# app.run(mode='inline')
server = app.server  # <-- important for deployment!

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)


# In[ ]:




