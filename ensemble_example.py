from ensemble import *
from errors import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
        Demonstrates standalone usage of the ensemble method.
    """
    # create the ensemble
    ensemble = Ensemble(train_style='overlap', base_size=400,
                        trainsize=500, num_segments=5, dataset='m', load_train='l')

    # load data and create datasets
    ensemble.set_data_from_file('./data/mackey.csv')
    ensemble.create_datasets()

    # Creates the methods, in this case Keras models
    ensemble.create_methods(batch_size=200, epochs=700, verbose=2, params='m')

    # trains or loads the model weight
    ensemble.train_methods()

    # get predictions and calculate errors
    testy = ensemble.testy
    predictions = ensemble.get_predictions(adaptive=True)
    print("MSE: ", mse(testy.tolist(), predictions))
    print("MAE: ", mae(testy.tolist(), predictions))

    # Save the weights to be used later
    # ensemble.save_method_weights('./weights/eurusd/eurusd')

    # plot the actual and predicted values
    plt.plot(testy, label='actual')
    plt.plot(predictions, label='predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Mackey-Glass Equation Output')
    plt.title('Actual vs. ADLE Predicted Mackey-Glass Equation Test Set')
    plt.legend()
    # plt.savefig('./graphs/results/adle_results_mackey.eps', dpi=1000)
    plt.show()
