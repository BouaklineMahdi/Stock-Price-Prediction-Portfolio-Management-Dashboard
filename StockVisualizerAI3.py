import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to send alerts
def send_alert(stock, price):
    sender_email = "bouaklinemahdi@gmail.com"
    receiver_email = "bouaklinemahdi@gmail.com"
    password = "Mahdi0610123"

    message = MIMEMultipart("alternative")
    message["Subject"] = f"Stock Alert: {stock}"
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"The stock {stock} has reached a price of {price}"
    part = MIMEText(text, "plain")
    message.attach(part)

    with smtplib.SMTP_SSL("smtp.example.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# List of top stocks
top_stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google (Alphabet)": "GOOGL",
    "Facebook (Meta)": "META",
    "Tesla": "TSLA",
    "Berkshire Hathaway": "BRK-B",
    "NVIDIA": "NVDA",
    "JPMorgan Chase": "JPM",
    "Johnson & Johnson": "JNJ"
}

# Fetch and process data
def fetch_data():
    stock_data = {}
    average_returns = {}
    
    for company, stock in top_stocks.items():
        df = yf.download(tickers=stock, period='3mo', interval='1d')
        if not df.empty:
            stock_data[stock] = df
            average_returns[stock] = df['Close'].pct_change().mean()
    
    return stock_data, average_returns

# Train AI model
def train_model(stock_data, stock):
    df = stock_data[stock]
    df['Date'] = df.index
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    prediction_days = 60
    x_train, y_train = [], []
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    
    return model, scaler

# Make predictions
def make_prediction(model, scaler, stock_data, stock):
    df = stock_data[stock]
    test_data = df[['Close']].values[-60:]
    test_data = scaler.transform(test_data)
    
    x_test = [test_data]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price[0][0]

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Stock Price Prediction Dashboard"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Stock:"),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[{'label': company, 'value': stock} for company, stock in top_stocks.items()],
                value='AAPL'
            ),
        ], width=4),
        dbc.Col([
            html.Label("Add to Portfolio:"),
            dcc.Input(id='portfolio-stock', type='text', placeholder='Enter stock ticker', debounce=True),
            html.Button(id='add-to-portfolio-button', n_clicks=0, children='Add to Portfolio'),
            html.H3(id='portfolio-output')
        ], width=4),
        dbc.Col([
            html.Button(id='alert-button', n_clicks=0, children='Send Alert')
        ], width=4)
    ]),
    dcc.Graph(id='stock-graph'),
    html.H3(id='prediction-output')
])

# Portfolio storage
portfolio = []

# Callback to update graph and prediction
@app.callback(
    [Output('stock-graph', 'figure'),
     Output('prediction-output', 'children'),
     Output('portfolio-output', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('add-to-portfolio-button', 'n_clicks'),
     Input('portfolio-stock', 'value'),
     Input('alert-button', 'n_clicks')]
)
def update_output(stock, portfolio_clicks, portfolio_stock, alert_clicks):
    stock_data, average_returns = fetch_data()
    model, scaler = train_model(stock_data, stock)
    predicted_price = make_prediction(model, scaler, stock_data, stock)
    
    df = stock_data[stock]
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'], name='Market Data'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA 20', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA 50', line=dict(color='orange', width=2)))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.3))
    
    fig.update_layout(
        title=f'{stock} Live Share Price and Analysis',
        yaxis_title='Stock Price (USD per Shares)',
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        xaxis_title='Time',
        legend=dict(x=0, y=1.1, orientation='h')
    )
    
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    if portfolio_clicks > 0 and portfolio_stock:
        if portfolio_stock not in portfolio:
            portfolio.append(portfolio_stock)
    
    if alert_clicks > 0:
        send_alert(stock, predicted_price)
    
    portfolio_display = f'Portfolio: {", ".join(portfolio)}'
    
    return fig, f'Predicted closing price for the next day: ${predicted_price:.2f}', portfolio_display

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
