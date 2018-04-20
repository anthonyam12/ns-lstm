from ensemble import *
from errors import *

if __name__ == '__main__':
    """
        Demonstrates standalone usage of the ensemble method.
    """
    ensemble = Ensemble(train_style='overlap', base_size=400,
                        trainsize=500, num_segments=5)
    ensemble.set_data_from_file('./data/mackey.csv')
    ensemble.create_datasets()
    ensemble.create_methods(batch_size=100, epochs=900, verbose=2, params='m')
    ensemble.train_methods()
    testy = ensemble.testy
    predictions = ensemble.get_predictions(adaptive=True)
    print("MSE: ", mse(testy.tolist(), predictions))
    print("MAE: ", mae(testy.tolist(), predictions))


## TODO: Get plots of predicted vs actual for LSTM, ADLE, and ARIMA
## TODO: Add to paper proof that data is nonstationayr (dickey fuller..)
