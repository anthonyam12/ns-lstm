from ensemble import *

if __name__ == '__main__':
    ensemble = Ensemble(train_style='overlap', base_size=250,
                        trainsize=500, num_segments=5)
    ensemble.set_data_from_file('./data/mackey.csv')
    ensemble.create_datasets()
    ensemble.create_methods(batch_size=100, epochs=700, verbose=2, params='m')
    ensemble.train_methods()
    print(ensemble.get_mse_from_predictions(adaptive=True))
