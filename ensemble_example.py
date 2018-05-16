from ensemble import *
from errors import *

import numpy as np

if __name__ == '__main__':
    """
        Demonstrates standalone usage of the ensemble method.
    """
    ensemble = Ensemble(train_style='overlap', base_size=1000,
                        trainsize=2000, num_segments=5)
    ensemble.set_data_from_file('./data/EURUSD.csv')
    ensemble.create_datasets()
    ## Try small batch later on
    ensemble.create_methods(batch_size=200, epochs=1050, verbose=2, params='e')
    ensemble.train_methods()
    testy = ensemble.testy
    predictions = ensemble.get_predictions(adaptive=True)
    print("MSE: ", mse(testy.tolist(), predictions))
    print("MAE: ", mae(testy.tolist(), predictions))
    print("R^2: ", r_squared(testy, np.asarray(predictions)))


## TODO: Get plots of predicted vs actual for LSTM, ADLE, and ARIMA
## TODO: Add to paper proof that data is nonstationayr (dickey fuller..)
