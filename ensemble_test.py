from ensemble import *

if __name__ == '__main__':
    ensemble = Ensemble(train_style='overlap', base_size=1000)
    ensemble.set_data_from_file('./data/Sunspots.csv')
    ensemble.create_datasets()
    ensemble.create_methods(batch_size=200, epochs=1000)
    ensemble.train_methods()
    ensemble.get_mse_from_predictions()
