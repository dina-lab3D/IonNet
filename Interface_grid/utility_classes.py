import numpy as np
import tensorflow.keras
import h5py

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, built using tutorial from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    'data shape is hardcoded to fit the current model being worked on which here is MGclassifier.'
    def __init__(self, list_IDs, labels, batch_size=128, dim=(32,32,32), n_channels=16,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        IDs are just the path names to the batches in the database
        and in total there will only be two possible labels for each batch which is either negative or positive
        :param list_IDs_temp:
        :return:
        """
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            with h5py.File(ID, "r") as f:
                a_group_key = list(f.keys())[0]
                X[i,] = f.get(a_group_key).value.reshape(*self.dim, self.n_channels)

            # Store class
            y[i] = self.labels[ID]

        return X, y


def load_dataset(positive_dataset_path="/cs/labs/dina/punims/Databases/Database_1_MG_RNA/MG_positive_resized/mg_positive_sample0.h5", negative_dataset_path="/cs/labs/dina/punims/Databases/Database_1_MG_RNA/MG_negative_resized/mg_negative_sample0.h5"):
    """
    loads the dataset as it was saved in premade random batches, may need to load multiple batches and randomize them
    data's current shape is 128,32,32,32,16
    :return: training and test data with the correct labels.
    """
    with h5py.File(positive_dataset_path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        positive_samples = list(f[a_group_key])
        if positive_samples:
            positive_samples = np.array(positive_samples)
            print(positive_samples.shape)

    with h5py.File(negative_dataset_path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        negative_samples = list(f[a_group_key])
        if negative_samples:
            negative_samples = np.array(negative_samples)
            print(negative_samples.shape)

    X_pos = positive_samples
    Y_pos = np.array([1]*X_pos.shape[0])
    X_neg = negative_samples
    Y_neg = np.array([0]*negative_samples.shape[0])
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((Y_pos, Y_neg))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test