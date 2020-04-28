import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_file_x, data_file_y, data_file_weights=None, batch_size=32,
                 in_shape=(256, 320, 4), out_shape=(256, 320, 1), shuffle=True):
        'Initialization'
        self.data_x = np.load(data_file_x, mmap_mode='r')
        self.data_y = np.load(data_file_y, mmap_mode='r')
        self.data_weights = None
        if data_file_weights is not None:
            self.data_weights = np.load(data_file_weights, mmap_mode='r')

        self.batch_size = batch_size
        self.list_IDs = np.arange(self.data_x.shape[0])
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, weights = self.__data_generation(list_IDs_temp)

        # Pad input to UNet shape
        padded_X = np.zeros((self.batch_size, 256, 320, 4))
        padded_X[:, 8:-8, :, :] = X

        # Add weights to labels if necessary
        if self.data_weights is not None:
            padded_y = np.zeros((self.batch_size, 256, 320, 2))
            padded_y[:, 8:-8, :, 1] = weights[:, :, :, 0]
        else:
            padded_y = np.zeros((self.batch_size, 256, 320, 1))

        padded_y[:, 8:-8, :, 0] = y[:, :, :, 0]
        padded_y[:, 8:-8, :, 0] = padded_y[:, 8:-8, :, 0] - 1
        return padded_X, padded_y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 240, 320, 4))
        y = np.empty((self.batch_size, 240, 320, 1))
        weights = np.empty((self.batch_size, 240, 320, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.data_x[ID]
            y[i] = self.data_y[ID]
            if self.data_weights is not None:
                weights[i] = self.data_weights[ID]

        return X, y, weights
