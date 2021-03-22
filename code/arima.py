from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

def arima_adtest(data):
    dftest = adfuller(data, autolag='AIC')
    print("ADF: ", dftest[0])
    print("P-Value: ", dftest[1])
    print("Num of Lags: ", dftest[2])
    print("Num of observations used for ADF regression and Critical values Calculation: ", dftest[3])
    print("Critical Values: ", )
    for key, val in dftest[4].items():
        print("\t", key, ":", val)

def arima_parameter_selection(data):
    stepwise_fit = auto_arima(data, trace=True, suppress_warnings=True)
    stepwise_fit.summary()

def arima_model(data):
    model = ARIMA(data, order=(1, 0, 0))
    model = model.fit()
    model.summary()
    start = len(data)
    end = len(data) + 6
    pred = model.predict(start=start, end=end, typ='levels')
    return pred.tolist()


