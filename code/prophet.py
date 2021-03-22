from fbprophet import Prophet
from matplotlib import pyplot as plt
from neuralprophet import NeuralProphet

def prophet(data):
    m = Prophet(interval_width=0.95, daily_seasonality=True)
    m.fit(data)
    future = m.make_future_dataframe(periods=7, freq='D')
    forecast = m.predict(future)
    forecast.tail(7)
    plot_prophet(forecast, m)
    return forecast.iloc[-7:, -1].tolist()

def plot_prophet(forecast, m):
    m.plot(forecast)
    m.plot_components(forecast)

def neural_prophet(data):
    m = NeuralProphet()
    m.fit(data, freq='D', epochs=1000)
    future = m.make_future_dataframe(data, periods=7)
    forecast = m.predict(future)
    plot_neural_prophet(forecast, m)
    return forecast.iloc[:, 2].tolist()

def plot_neural_prophet(forecast, m):
    m.plot(forecast)
    m.plot_components(forecast)