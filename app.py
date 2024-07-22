import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from textblob import TextBlob

# Load the data from CSV
file_path = 'E-commerce_Data.csv'  # Adjust the path if needed
df = pd.read_csv(file_path)

# Ensure 'date' is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Calculate the time periods
df['day'] = df['date'].dt.date
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
df['month'] = df['date'].dt.to_period('M').apply(lambda r: r.start_time)

# Calculate bounce rates, total visits, average session duration, and conversion rates
daily_stats = df.groupby('date').agg({'bounce': 'mean', 'session_id': 'count', 'time_spent': 'mean', 'purchase': 'mean'}).reset_index().rename(columns={'session_id': 'total_visits', 'time_spent': 'avg_session_duration', 'purchase': 'conversion_rate'})
weekly_stats = df.groupby('week').agg({'bounce': 'mean', 'session_id': 'count', 'time_spent': 'mean', 'purchase': 'mean'}).reset_index().rename(columns={'session_id': 'total_visits', 'time_spent': 'avg_session_duration', 'purchase': 'conversion_rate'})
monthly_stats = df.groupby('month').agg({'bounce': 'mean', 'session_id': 'count', 'time_spent': 'mean', 'purchase': 'mean'}).reset_index().rename(columns={'session_id': 'total_visits', 'time_spent': 'avg_session_duration', 'purchase': 'conversion_rate'})

# Calculate KPIs for the last week
last_week_date = df['date'].max() - pd.Timedelta(days=7)
last_week_data = df[df['date'] > last_week_date]
kpi_bounce_rate = last_week_data['bounce'].mean()
kpi_conversion_rate = last_week_data['purchase'].mean()
kpi_session_duration = last_week_data['time_spent'].mean()
kpi_rating = last_week_data["rating"].mean()

# Calculate funnel metrics
funnel_metrics = df['funnel_stage'].value_counts().reindex(['Homepage', 'Product Page', 'Cart', 'Checkout', 'Purchase'], fill_value=0).reset_index()
funnel_metrics.columns = ['stage', 'count']
funnel_metrics['percent_initial'] = funnel_metrics['count'] / funnel_metrics['count'].iloc[0] * 100
funnel_metrics['percent_previous'] = funnel_metrics['count'].div(funnel_metrics['count'].shift(1).replace(0, np.nan)) * 100


# Perform sentiment analysis
df['sentiment'] = df['user_comments'].apply(lambda comment: TextBlob(comment).sentiment.polarity)
df['sentiment_label'] = df['sentiment'].apply(lambda polarity: 'Positive' if polarity > 0 else ('Negative' if polarity < 0 else 'Neutral'))

#number of purchaces
daily_purchases = df[df['purchase'] == 1].groupby('day').size().reset_index(name='purchases')
weekly_purchases = df[df['purchase'] == 1].groupby('week').size().reset_index(name='purchases')
monthly_purchases = df[df['purchase'] == 1].groupby('month').size().reset_index(name='purchases')


# Aggregate rating data for daily, weekly, and monthly
daily_ratings = df.groupby('day')['rating'].mean().reset_index(name='average_rating')
weekly_ratings = df.groupby('week')['rating'].mean().reset_index(name='average_rating')
monthly_ratings = df.groupby('month')['rating'].mean().reset_index(name='average_rating')


# Calculate new and returning customers
df['is_new_customer'] = ~df.duplicated(subset=['user_id'], keep='first')
daily_customers = df.groupby('day').agg(new_customers=('is_new_customer', 'sum'), total_customers=('user_id', 'nunique')).reset_index()
weekly_customers = df.groupby('week').agg(new_customers=('is_new_customer', 'sum'), total_customers=('user_id', 'nunique')).reset_index()
monthly_customers = df.groupby('month').agg(new_customers=('is_new_customer', 'sum'), total_customers=('user_id', 'nunique')).reset_index()

# Calculate churn rate
daily_customers['churn_rate'] = (daily_customers['total_customers'] - daily_customers['new_customers']) / daily_customers['total_customers']
weekly_customers['churn_rate'] = (weekly_customers['total_customers'] - weekly_customers['new_customers']) / weekly_customers['total_customers']
monthly_customers['churn_rate'] = (monthly_customers['total_customers'] - monthly_customers['new_customers']) / monthly_customers['total_customers']


# Aggregate revenue data for daily, weekly, and monthly
daily_revenue = df.groupby('day')['amount_spent'].sum().reset_index(name='revenue')
weekly_revenue = df.groupby('week')['amount_spent'].sum().reset_index(name='revenue')
monthly_revenue = df.groupby('month')['amount_spent'].sum().reset_index(name='revenue')

# Calculate returning customers
daily_customers['returning_customers'] = daily_customers['total_customers'] - daily_customers['new_customers']
weekly_customers['returning_customers'] = weekly_customers['total_customers'] - weekly_customers['new_customers']
monthly_customers['returning_customers'] = monthly_customers['total_customers'] - monthly_customers['new_customers']



# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("E-commerce Metrics Dashboard", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    html.P(
        "Welcome to the E-commerce Metrics Dashboard. This dashboard provides a comprehensive view of key performance indicators (KPIs)" 
        "for our e-commerce platform. You can explore various metrics such as bounce rates, total visits, average session duration,"
        "conversion rates, and purchases over different time intervals. Additionally, you can analyze user demographics and sentiment" 
        "analysis based on user comments. Use the tabs and filters to navigate and gain insights into the performance and user "
        "behavior on our platform.",
        style={'textAlign': 'justify', 'color': 'white'}
    ),
    html.P(
        "Designed by  Data Scientist Alfonso Cervantes B commissiond by NEVO business Center",
        style={'textAlign': 'justify', 'color': 'white'}
    ),
    # KPI sssssssssss
    html.H2("KPIs from Last Week Period", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    html.Div([
        html.Div([
            html.H3("Bounce Rate", style={'color': '#FFFFFF'}),
            html.P(f"{kpi_bounce_rate:.2%}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#FFFFFF'})
        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center', 'backgroundColor': '#4CAF50', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}),
        html.Div([
            html.H3("Conversion Rate", style={'color': '#FFFFFF'}),
            html.P(f"{kpi_conversion_rate:.2%}", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#FFFFFF'})
        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center', 'backgroundColor': '#4CAF50', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}),
        html.Div([
            html.H3("Average Session Duration", style={'color': '#FFFFFF'}),
            html.P(f"{kpi_session_duration:.2f} seconds", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#FFFFFF'})
        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center', 'backgroundColor': '#4CAF50', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}),
        html.Div([
            html.H3("Average rating", style={'color': '#FFFFFF'}),
            html.P(f"{kpi_rating:.2f}/5.0", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#FFFFFF'})
        ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center', 'backgroundColor': '#4CAF50', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'})
    ], style={'textAlign': 'center'}),
    ### ratings
    html.Div([
   
    html.H2("Ratings Over Time", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    dcc.Dropdown(
        id='time-interval-dropdown2',
        options=[
            {'label': 'Daily', 'value': 'daily'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'}
        ],
        value='daily'
    ),
    dcc.Graph(id='ratings-over-time')
    ], style={'backgroundColor': '#2E2E2E', 'padding': '20px'}),
    ### TIME EVOLUTION METRICS
    html.H2("Time evolution of key metrics", style={'textAlign': 'center', 'color': 'white'}),   
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Daily', children=[
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='daily-bounce-graph',
                        figure=px.line(daily_stats, x='date', y='bounce', title='Daily Bounce Rate', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='daily-visits-graph',
                        figure=px.line(daily_stats, x='date', y='total_visits', title='Daily Total Visits', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='daily-session-duration-graph',
                        figure=px.line(daily_stats, x='date', y='avg_session_duration', title='Daily Average Session Duration', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='daily-conversion-rate-graph',
                        figure=px.line(daily_stats, x='date', y='conversion_rate', title='Daily Conversion Rate', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ])
        ]),
        dcc.Tab(label='Weekly', children=[
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='weekly-bounce-graph',
                        figure=px.line(weekly_stats, x='week', y='bounce', title='Weekly Bounce Rate', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='weekly-visits-graph',
                        figure=px.line(weekly_stats, x='week', y='total_visits', title='Weekly Total Visits', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='weekly-session-duration-graph',
                        figure=px.line(weekly_stats, x='week', y='avg_session_duration', title='Weekly Average Session Duration', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='weekly-conversion-rate-graph',
                        figure=px.line(weekly_stats, x='week', y='conversion_rate', title='Weekly Conversion Rate', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ])
        ]),
        dcc.Tab(label='Monthly', children=[
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='monthly-bounce-graph',
                        figure=px.line(monthly_stats, x='month', y='bounce', title='Monthly Bounce Rate', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='monthly-visits-graph',
                        figure=px.line(monthly_stats, x='month', y='total_visits', title='Monthly Total Visits', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='monthly-session-duration-graph',
                        figure=px.line(monthly_stats, x='month', y='avg_session_duration', title='Monthly Average Session Duration', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='monthly-conversion-rate-graph',
                        figure=px.line(monthly_stats, x='month', y='conversion_rate', title='Monthly Conversion Rate', template='plotly_dark')
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ])
        ]),
        dcc.Tab(label='Funnel', children=[
        html.Div([
            dcc.Graph(
                id='funnel-graph',
                figure=go.Figure(go.Funnel(
                    y=funnel_metrics['stage'],
                    x=funnel_metrics['count'],
                    textinfo='value+percent initial+percent previous'
                )).update_layout(title='Funnel Chart', template='plotly_dark')
            )
        ])
])
    ]),
html.Div([
    html.H2("Demographic distribution", style={'textAlign': 'center', 'color': 'white'}),   
    html.Div([
        html.Div([
            html.Label('Gender',style={'color': 'white'}),
            dcc.Dropdown(
                id='gender-filter',
                options=[{'label': gender, 'value': gender} for gender in df['gender'].unique()],
                value=None,
                multi=True,
                placeholder='Select Gender'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Country',style={'color': 'white'}),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': country, 'value': country} for country in df['country'].unique()],
                value=None,
                multi=True,
                placeholder='Select Country'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Age Range',style={'color': 'white'}),
            dcc.RangeSlider(
                id='age-slider',
                min=df['age'].min(),
                max=df['age'].max(),
                value=[df['age'].min(), df['age'].max()],
                marks={i: str(i) for i in range(df['age'].min(), df['age'].max() + 1, 5)}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={}),

    html.Div([
        html.Div([
            dcc.Graph(id='age-distribution')
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='gender-distribution')
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='country-distribution')
        ], style={'width': '33%', 'display': 'inline-block'})
    ])
], style={}),
## Sentiment analysis
html.Div([
    html.H2("Sentiment Analysis of User Comments",style={"color":"white",'textAlign': 'center'}),
     
    dcc.Dropdown(
        id='sentiment-dropdown',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Positive', 'value': 'Positive'},
            {'label': 'Neutral', 'value': 'Neutral'},
            {'label': 'Negative', 'value': 'Negative'}
        ],
        value='All'
    ),
    
    dcc.Graph(id='sentiment-pie-chart'),
    
    dcc.Graph(id='sentiment-age-chart'),
    
    
    dcc.Graph(id='sentiment-gender-chart'),
    
    
    dcc.Graph(id='sentiment-country-chart')
       
]),
### purchaces
html.Div([
    html.H2("Purchases Over Time",style={"color":"white",'textAlign': 'center'}),
    dcc.Dropdown(
        id='time-interval-dropdown',
        options=[
            {'label': 'Daily', 'value': 'daily'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'}
        ],
        value='daily'
    ),
    dcc.Graph(id='purchases-over-time')
]),

# Customer churn dashboard
html.Div([
    
    html.H2("Customer Churn Rate Over Time", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    dcc.Dropdown(
        id='time-interval-dropdown-churn',
        options=[
            {'label': 'Daily', 'value': 'daily'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'}
        ],
        value='daily'
    ),
    dcc.Graph(id='churn-rate-over-time')
], style={'backgroundColor': '#2E2E2E', 'padding': '20px'}),

# revenue
html.Div([
    
    html.H2("Revenue Over Time", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    dcc.Dropdown(
        id='time-interval-dropdown-revenue',
        options=[
            {'label': 'Daily', 'value': 'daily'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'}
        ],
        value='daily'
    ),
    dcc.Graph(id='revenue-over-time')
], style={'backgroundColor': '#2E2E2E', 'padding': '20px'}),

#customer return
html.Div([
    
    html.H2("Customer Retention Over Time", style={'textAlign': 'center', 'color': '#FFFFFF'}),
    dcc.Dropdown(
        id='time-interval-dropdown-retention',
        options=[
            {'label': 'Daily', 'value': 'daily'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'}
        ],
        value='daily'
    ),
    dcc.Graph(id='retention-over-time')
], style={'backgroundColor': '#2E2E2E', 'padding': '20px'})

], style={'backgroundColor': '#2E2E2E', 'padding': '20px','marginLeft': '5%', 'marginRight': '5%'})



# Define the callbacks to update the graphs
@app.callback(
    [Output('age-distribution', 'figure'),
      Output('gender-distribution', 'figure'),
      Output('country-distribution', 'figure')],
    [Input('gender-filter', 'value'),
      Input('country-filter', 'value'),
      Input('age-slider', 'value')]
)

def update_graphs(selected_genders, selected_countries, selected_age_range):
    # Filter data based on selections
    filtered_data = df[
        (df['age'] >= selected_age_range[0]) &
        (df['age'] <= selected_age_range[1])
    ]
    if selected_genders:
        filtered_data = filtered_data[filtered_data['gender'].isin(selected_genders)]
    if selected_countries:
        filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]
    
    # Age distribution
    age_fig = px.histogram(filtered_data, x='age', nbins=20, title='Age Distribution')
    age_fig.update_layout(
        template='plotly_dark',
        xaxis_title='Age',
        yaxis_title='Count',
        annotations=[dict(
            text='Distribution of user ages',
            xref='paper', yref='paper',
            x=0.5, y=1.1, showarrow=False
        )]
    )

    # Gender distribution
    gender_fig = px.pie(filtered_data, names='gender', title='Gender Distribution')
    gender_fig.update_layout(
        template='plotly_dark',
        annotations=[dict(
            text='Proportion of users by gender',
            xref='paper', yref='paper',
            x=0.5, y=1.1, showarrow=False
        )]
    )

    # Country distribution
    country_counts = filtered_data['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_fig = px.bar(country_counts, x='country', y='count', title='Country Distribution')
    country_fig.update_layout(
        template='plotly_dark',
        xaxis_title='Country',
        yaxis_title='Count',
        annotations=[dict(
            text='Number of users by country',
            xref='paper', yref='paper',
            x=0.5, y=1.1, showarrow=False
        )]
    )

    return age_fig, gender_fig, country_fig


# Define callback to update the sentiment pie chart
@app.callback(
    Output('sentiment-pie-chart', 'figure'),
    [Input('sentiment-dropdown', 'value')]
)
def update_sentiment_pie_chart(sentiment):
    if sentiment == 'All':
        filtered_data = df
    else:
        filtered_data = df[df['sentiment_label'] == sentiment]
    
    pie_fig = px.pie(filtered_data, names='sentiment_label', title='Sentiment Proportion', template='plotly_dark')
    pie_fig.update_traces(textinfo='percent+label')
    return pie_fig


# Define callback to update the sentiment by age chart
@app.callback(
    Output('sentiment-age-chart', 'figure'),
    [Input('sentiment-dropdown', 'value')]
)
def update_sentiment_age_chart(sentiment):
    if sentiment == 'All':
        filtered_data = df
    else:
        filtered_data = df[df['sentiment_label'] == sentiment]
    
    age_fig = px.histogram(filtered_data, x='age', color='sentiment_label', title='Sentiment Proportion by Age', barmode='group', template='plotly_dark')
    age_fig.update_layout(legend_title_text='Sentiment')
    return age_fig

# Define callback to update the sentiment by gender chart
@app.callback(
    Output('sentiment-gender-chart', 'figure'),
    [Input('sentiment-dropdown', 'value')]
)
def update_sentiment_gender_chart(sentiment):
    if sentiment == 'All':
        filtered_data = df
    else:
        filtered_data = df[df['sentiment_label'] == sentiment]
    
    gender_fig = px.histogram(filtered_data, x='gender', color='sentiment_label', title='Sentiment Proportion by Gender', barmode='group', template='plotly_dark')
    gender_fig.update_layout(legend_title_text='Sentiment')
    return gender_fig


# Define callback to update the sentiment by country chart
@app.callback(
    Output('sentiment-country-chart', 'figure'),
    [Input('sentiment-dropdown', 'value')]
)
def update_sentiment_country_chart(sentiment):
    if sentiment == 'All':
        filtered_data = df
    else:
        filtered_data = df[df['sentiment_label'] == sentiment]
    
    country_fig = px.histogram(filtered_data, x='country', color='sentiment_label', title='Sentiment Proportion by Country', barmode='group', template='plotly_dark')
    country_fig.update_layout(legend_title_text='Sentiment')
    return country_fig

# Define callback to update the purchases over time chart
@app.callback(
    Output('purchases-over-time', 'figure'),
    [Input('time-interval-dropdown', 'value')]
)
def update_purchases_over_time(interval):
    if interval == 'daily':
        time_data = daily_purchases
        x_column = 'day'
        title = 'Daily Purchases Over Time'
    elif interval == 'weekly':
        time_data = weekly_purchases
        x_column = 'week'
        title = 'Weekly Purchases Over Time'
    else:
        time_data = monthly_purchases
        x_column = 'month'
        title = 'Monthly Purchases Over Time'
    
    time_fig = px.line(time_data, x=x_column, y='purchases', title=title, template='plotly_dark')
    return time_fig

# Define callback to update the ratings over time chart
@app.callback(
    Output('ratings-over-time', 'figure'),
    [Input('time-interval-dropdown2', 'value')]
)
def update_ratings_over_time(interval):
    if interval == 'daily':
        time_data = daily_ratings
        x_column = 'day'
        title = 'Daily Average Rating Over Time'
    elif interval == 'weekly':
        time_data = weekly_ratings
        x_column = 'week'
        title = 'Weekly Average Rating Over Time'
    else:
        time_data = monthly_ratings
        x_column = 'month'
        title = 'Monthly Average Rating Over Time'
    
    time_fig = px.line(time_data, x=x_column, y='average_rating', title=title, template='plotly_dark')
    return time_fig

# Define callback to update the churn rate chart
@app.callback(
    Output('churn-rate-over-time', 'figure'),
    [Input('time-interval-dropdown-churn', 'value')]
)
def update_churn_rate_over_time(interval):
    if interval == 'daily':
        time_data = daily_customers
        x_column = 'day'
        title = 'Daily Customer Churn Rate Over Time'
    elif interval == 'weekly':
        time_data = weekly_customers
        x_column = 'week'
        title = 'Weekly Customer Churn Rate Over Time'
    else:
        time_data = monthly_customers
        x_column = 'month'
        title = 'Monthly Customer Churn Rate Over Time'
    
    time_fig = px.line(time_data, x=x_column, y='churn_rate', title=title, template='plotly_dark')
    return time_fig

# Define callback to update the revenue over time chart
@app.callback(
    Output('revenue-over-time', 'figure'),
    [Input('time-interval-dropdown-revenue', 'value')]
)
def update_revenue_over_time(interval):
    if interval == 'daily':
        time_data = daily_revenue
        x_column = 'day'
        title = 'Daily Revenue Over Time'
    elif interval == 'weekly':
        time_data = weekly_revenue
        x_column = 'week'
        title = 'Weekly Revenue Over Time'
    else:
        time_data = monthly_revenue
        x_column = 'month'
        title = 'Monthly Revenue Over Time'
    
    time_fig = px.line(time_data, x=x_column, y='revenue', title=title, template='plotly_dark')
    return time_fig


# Define callback to update the retention over time chart
@app.callback(
    Output('retention-over-time', 'figure'),
    [Input('time-interval-dropdown-retention', 'value')]
)
def update_retention_over_time(interval):
    if interval == 'daily':
        time_data = daily_customers
        x_column = 'day'
        title = 'Daily Customer Retention Over Time'
    elif interval == 'weekly':
        time_data = weekly_customers
        x_column = 'week'
        title = 'Weekly Customer Retention Over Time'
    else:
        time_data = monthly_customers
        x_column = 'month'
        title = 'Monthly Customer Retention Over Time'
    
    time_fig = px.line(time_data, x=x_column, y=['new_customers', 'returning_customers'], title=title, template='plotly_dark')
    return time_fig



if __name__ == '__main__':
    app.run_server(debug=True)
