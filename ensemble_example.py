from ensemble import *
from errors import *

if __name__ == '__main__':
    """
        Demonstrates standalone usage of the ensemble method.
    """
    # create the ensemble
    ensemble = Ensemble(train_style='overlap', base_size=1000,
                        trainsize=2000, num_segments=5, dataset='s', load_train='t')

    # load data and create datasets
    ensemble.set_data_from_file('./data/Sunspots.csv')
    ensemble.create_datasets()

    # Creates the methods, in this case Keras models
    ensemble.create_methods(batch_size=200, epochs=700, verbose=2, params='s')

    # trains or loads the model weights
    ensemble.train_methods()

    # get predictions and calculate errors
    testy = ensemble.testy
    predictions = ensemble.get_predictions(adaptive=True)
    print("MSE: ", mse(testy.tolist(), predictions))
    print("MAE: ", mae(testy.tolist(), predictions))

    # Save the weights to be used later
    # ensemble.save_method_weights('./weights/eurusd/eurusd')
