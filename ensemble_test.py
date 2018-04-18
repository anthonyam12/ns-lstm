from ensemble import *

if __name__ == '__main__':
    ensemble = Ensemble(train_style='overlap', base_size=1300,
                        trainsize=2000, num_segments=5)
    ensemble.set_data_from_file('./data/EURUSD.csv')
    ensemble.create_datasets()
    ensemble.create_methods(batch_size=200, epochs=400, verbose=2)
    ensemble.train_methods()
    print(ensemble.get_mse_from_predictions(adaptive=True, window_size=400))
