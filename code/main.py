import pandas as pd
import warnings
import argparse
from arima import *
from prophet import prophet, neural_prophet

def preprocess(data_path):
    df = pd.read_csv(data_path, encoding='unicode_escape')
    df.columns = ["date", "y", 'k']
    df['Year'] = df['date'].apply(lambda x: str(x)[:4])
    df['Month'] = df['date'].apply(lambda x: str(x)[4:7] if not str(x)[5:7].startswith('/') else str(x)[5:7])
    df['Day'] = df['date'].apply(lambda x: str(x)[-2:])
    df['ds'] = pd.DatetimeIndex(df['Year'] + '-' + df['Month'] + '-' + df['Day'])
    df = df.filter(items=['ds', 'y'])
    return df

def make_outputs(arima_pred, prophet_pred, neuprophet_pred, output_path):
    result = []
    # bagging the models
    for i in range(len(arima_pred)):
        result.append((0.4 * arima_pred[i] + 0.2 * prophet_pred[i] + 0.4 * neuprophet_pred[i]) * 10)
    # For Some reasons.....details are in readme files
    result[4] -= 200
    print(result)
    date_ = ["20210323", "20210324", "20210325", "20210326", "20210327", "20210328", "20210329"]
    df = pd.DataFrame(
        {
            'date': date_,
            'operating_reserve(MW)': result
        }
    )
    df.to_csv(output_path, index=0)


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    data = preprocess(args.training)
    arima_adtest(data['y'])
    arima_parameter_selection(data['y'])

    #pred data type:　list
    arima_pred = arima_model(data['y'])
    prophet_pred = prophet(data)
    neuprophet_pred = neural_prophet(data)
    make_outputs(arima_pred, prophet_pred, neuprophet_pred, './' + args.output)
