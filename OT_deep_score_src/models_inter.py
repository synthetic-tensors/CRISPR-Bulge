"""
This module contains the models interfaces and some required functions
"""
from abc import ABC, abstractmethod

import pickle
import shutil
import uuid

from OT_deep_score_src.general_utilities import Model_task, SEED, FILES_DIR

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


def validate_model_task(model_task):
    """
    Validates whether a given model task is supported.

    Args:
        model_task (Model_task): The model task to validate.

    Raises:
        ValueError: If the model task is not "CLASSIFICATION_TASK" or "REGRESSION_TASK".
    """
    if model_task not in [Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK]:
        raise ValueError("Model task should be one of {}".format(
            [Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK]))


def split_to_train_and_val(x, y, sample_weight, val_size, validation_data, stratify_in_split):
    """
    Splits input data into training and validation sets, optionally using stratification.
    Handles cases where validation data is provided or needs to be generated.

    Args:
        x: Input data (can be a single array or a list of arrays).
        y: Target labels.
        sample_weight (array-like, optional): Sample weights.
        val_size (float, optional): Proportion of data to use for validation (if validation_data is None).
        validation_data (tuple of (x_val, y_val), optional): Pre-existing validation data.
        stratify_in_split (bool): If True, uses stratification during the split.

    Returns:
        tuple: (x, y, sample_weight, validation_data) where:
            - x: Training input data.
            - y: Training labels.
            - sample_weight: Training sample weights (if provided).
            - validation_data: Validation data (generated if not provided).
    """
    if val_size is not None and validation_data is None:
        # split weight and set stratify
        # TODO: I added stratify, So I might need to test its effect
        if sample_weight is not None:
            print("stratify_in_split:", stratify_in_split)
            stratify = sample_weight if stratify_in_split else None
            sample_weight, sample_weight_val = train_test_split(
                sample_weight, test_size=val_size, random_state=SEED, stratify=stratify)
        else:
            sample_weight, sample_weight_val, stratify = None, None, None
        # split y
        y, y_val = train_test_split(y, test_size=val_size, random_state=SEED, stratify=stratify)
        # split x
        if isinstance(x, list):
            x_split = [train_test_split(
                x_item, test_size=val_size, random_state=SEED, stratify=stratify) for x_item in x]
            x = [x_item[0] for x_item in x_split]
            x_val = [x_item[1] for x_item in x_split]
        else:
            x, x_val = train_test_split(x, test_size=val_size, random_state=SEED, stratify=stratify)

        validation_data = (x_val, y_val) if sample_weight is None else (x_val, y_val, sample_weight_val)

    return x, y, sample_weight, validation_data


class Model(ABC):
    """
    An abstract base class defining the core interface for all models.
    Provides functionality for saving and loading model instances.
    """

    def __init__(self):
        """
        Initializes a Model instance.
        """
        self._model = None

    @property
    def model(self):
        """
        Returns the underlying model object.
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Sets the underlying model object.

        Args:
            model: The model object to be assigned. Can be an instance of "Model" or a compatible raw model.
        """
        self._model = model.model if isinstance(model, Model) else model

    @abstractmethod
    def construct(self):
        """
        Defines the model architecture.  (Abstract method)
        """
        pass

    @abstractmethod
    def fit(self, x, y):
        """
        Trains the model on given data. (Abstract method)

        Args:
            x: Input data.
            y: Target labels.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Generates predictions using the trained model.  (Abstract method)

        Args:
            x: Input data.

        Returns:
            Predictions from the model.
        """
        pass

    @abstractmethod
    def save(self, file_path_and_name):
        """
        Saves the model to a file.  (Abstract method)

        Args:
            file_path_and_name: The path and filename where the model should be saved.
        """
        pass

    @abstractmethod
    def load(self, file_path_and_name):
        """
        Loads the model from a file.  (Abstract method)

        Args:
            file_path_and_name: The path and filename of the saved model.
        """
        pass

    def save_model_instance(self, file_path_and_name):
        """
        Saves both the model object and the current instance of the "Model" class to file.

        Args:
            file_path_and_name (str):  The path and filename where the model should be saved.
                                       (automatically adds extensions).
        """
        self.save(file_path_and_name)
        # We do not want to save the model itself as a pickle file
        model = self.model
        self.model = None
        with open(file_path_and_name + ".pkl", "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.model = model

    @classmethod
    def load_model_instance(cls, file_path_and_name):
        """
        Loads a saved model instance (both the model object and the "Model" class instance).

        Args:
            file_path_and_name (str): The path and filename of the saved model.
                                      (automatically adds extensions).

        Returns:
            Model: The loaded model instance.
        """
        with open(file_path_and_name + ".pkl", "rb") as f:
            model_instance: Model = pickle.load(f)
        model_instance.load(file_path_and_name)

        return model_instance


class Sklearn_model(Model):
    """
    A class representing scikit-learn models, providing a consistent interface
    and handling model saving/loading.
    """

    def __init__(self, model_task):
        """
        Initializes an Sklearn_model instance.

        Args:
            model_task (Model_task): Specifies whether it is a "CLASSIFICATION_TASK" or "REGRESSION_TASK".
        """
        validate_model_task(model_task)
        self._model_task = model_task
        super().__init__()

    def fit(self, x, y, sample_weight=None):
        """
        Trains the scikit-learn model.

        Args:
            x: Input data.
            y: Target labels.
            sample_weight (optional): Weights for each sample.

        Raises:
            TypeError: If the internal model object (`self.model`) is not set.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, x):
        """
        Generates predictions using the trained scikit-learn model.

        Args:
            x: Input data.

        Returns:
            Predictions from the model. For classification tasks, returns class probabilities.
            For regression tasks, returns predicted values.

        Raises:
            TypeError: If the internal model object (`self.model`) is not set.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            return self.model.predict_proba(x)  # [:, 1]
        else:
            return self.model.predict(x)

    def save(self, file_path_and_name, suffix=".skl"):
        """
        Saves the scikit-learn model to a file.

        Args:
            file_path_and_name (str):  The path and filename where the model should be saved.
            suffix (str, optional):  File extension (default: ".skl").
        """
        with open(file_path_and_name + suffix, "wb") as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path_and_name, suffix=".skl"):
        """
        Loads a previously saved scikit-learn model from a file.

        Args:
            file_path_and_name (str):  The path and filename of the saved model.
            suffix (str, optional):  File extension (default: ".skl").
        """
        with open(file_path_and_name + suffix, "rb") as f:
            self.model = pickle.load(f)


def get_optimizer(optimizer_name, learning_rate):
    """
    Creates a TensorFlow optimizer instance based on its name and learning rate.

    Args:
        optimizer_name (str): The name of the optimizer class
            (e.g., "Adam", "SGD", "RMSprop").
        learning_rate (float): The desired learning rate for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: An instance of the specified TensorFlow optimizer,
            configured with the provided learning rate.
    """
    optimizer = tf.keras.optimizers.get(
            {"class_name": optimizer_name, "config": {"learning_rate": learning_rate}})

    return optimizer


class NN_model(Model):
    """
    A class representing neural network models built using TensorFlow.
    Provides functionality for model compilation, training, prediction, saving, and loading.
    """

    PREDICT_BATCH_SIZE = 4096  # batch size for prediction

    def __init__(self, batch_size, epochs):
        """
        Initializes an NN_model instance.

        Args:
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
        """
        tf.keras.backend.clear_session()  # clear the session to clear old layer names
        self._batch_size = batch_size
        self._epochs = epochs
        super().__init__()

    @abstractmethod
    def compile(self):
        """
        Compiles the neural network model.

        Defines the optimizer, loss function, and metrics used for training.
        (Abstract method).
        """
        pass

    @staticmethod
    def one_hot_to_categorical(x, validation_data=None):
        """
        Converts one-hot encoded data to categorical format. Works with multiple inputs.

        Args:
            x: Input data (can be a list if multiple inputs).
               If there are multiple inputs, then the first input is the one-hot.
            validation_data (optional): Validation data, adjusted if provided.

        Returns:
            tuple: Transformed input data and validation data (if provided).
        """
        if isinstance(x, list):
            x[0] = np.argmax(x[0], axis=2)
        else:
            x = np.argmax(x, axis=2)

        if validation_data is not None:
            validation_data = list(validation_data)
            if isinstance(validation_data[0], list):
                validation_data[0][0] = np.argmax(validation_data[0][0], axis=2)
            else:
                validation_data[0] = np.argmax(validation_data[0], axis=2)
            validation_data = tuple(validation_data)

        return x, validation_data

    @staticmethod
    def model_checkpoint_callbak(checkpoint_filepath, monitor="val_loss", mode="min"):
        """
        Creates a TensorFlow callback for saving the best model during training.

        Args:
            checkpoint_filepath (str): Base filepath where checkpoints will be saved.
            monitor (str, optional): Quantity to monitor (default: "val_loss").
            mode (str, optional): Mode for monitoring ("min" or "max", default: "min").

        Returns:
            tf.keras.callbacks.ModelCheckpoint: A TensorFlow callback object.
        """
        checkpoint_filepath = checkpoint_filepath + "/checkpoint"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            verbose=1,
            save_weights_only=True,
            monitor=monitor,
            mode=mode,
            save_best_only=True
        )

        return model_checkpoint_callback

    def fit(self, x, y, validation_data, sample_weight, val_size, verbose, choose_best_epoch,
            train_fit_log_path, early_stopping, stratify_in_split, use_tfrecords, **additional_fit_params):
        """
        Trains the neural network model.

        Args:
            x: Input data.
            y: Target labels.
            validation_data (tuple or None): Validation data.
            sample_weight (array-like, optional): Sample weights.
            val_size (float, optional): Proportion of data to use for validation if "validation_data" is not provided.
            verbose (int): Verbosity level during training.
            choose_best_epoch (bool): If True, loads the best weights based on validation loss (default: False).
            train_fit_log_path (str, optional): Path to save training logs in CSV format.
            early_stopping (bool): If True, uses early stopping to prevent overfitting (default: True).
            stratify_in_split (bool): If True, uses stratification when splitting data into
                                                train/validation sets.
            use_tfrecords (bool): If True, loads data from TFRecord format.
            **additional_fit_params: Additional keyword arguments to pass to "self.model.fit".
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        x, y, sample_weight, validation_data = split_to_train_and_val(
            x, y, sample_weight, val_size, validation_data, stratify_in_split)

        callbacks = additional_fit_params.pop("callbacks", [])
        # define checkpoint callback if needed
        checkpoint_filepath = FILES_DIR + "tmp" + str(uuid.uuid4().hex)  # create unique tmp folder name
        if early_stopping and validation_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True))
        if choose_best_epoch and validation_data is not None:
            callbacks.append(NN_model.model_checkpoint_callbak(checkpoint_filepath))
        # define CSVlogger callback of train if needed
        if train_fit_log_path is not None:
            callbacks.append(tf.keras.callbacks.CSVLogger(train_fit_log_path + "_train_log.csv"))
        if not callbacks:
            callbacks = None

        if use_tfrecords:
            dataset = TFRecord_dataset(x, y, sample_weight).dataset
        else:
            x_dataset = tf.data.Dataset.from_tensor_slices(tuple(x) if isinstance(x, list) else x)
            y_dataset = tf.data.Dataset.from_tensor_slices(y)
            if sample_weight is not None:
                sample_weight_dataset = tf.data.Dataset.from_tensor_slices(sample_weight)
                dataset = tf.data.Dataset.zip((x_dataset, y_dataset, sample_weight_dataset))
            else:
                dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

        dataset = dataset.batch(self._batch_size).prefetch(tf.data.AUTOTUNE)
        self.model.fit(dataset,
                       validation_data=validation_data,
                       epochs=self._epochs,
                       verbose=verbose,
                       callbacks=callbacks,
                       **additional_fit_params)

        if choose_best_epoch and validation_data is not None:
            self.model.load_weights(checkpoint_filepath + "/checkpoint")
            shutil.rmtree(checkpoint_filepath, ignore_errors=True)

        # this should force clear after fit
        tf.keras.backend.clear_session()

    def predict(self, x):
        """
        Generates predictions using the trained neural network model.

        Args:
            x: Input data.

        Returns:
            array-like: Predictions from the model.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        # TODO: make it flexible to other batch sizes
        return self.model.predict(x, batch_size=NN_model.PREDICT_BATCH_SIZE).squeeze()

    def save(self, file_path_and_name):
        """
        Saves the neural network model to an HDF5 file.

        Args:
            file_path_and_name (str):  The path and filename where the model should be saved.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.save(file_path_and_name + ".h5")

    def load(self, file_path_and_name):
        """
        Loads a previously saved neural network model from an HDF5 file.

        Args:
            file_path_and_name (str):  The path and filename of the saved model.
        """
        self.model = tf.keras.models.load_model(file_path_and_name + ".h5")


class TFRecord_dataset():
    """
    A class representing a dataset stored in TFRecord format. Provides
    functionality for encoding data into TFRecords and decoding the data back.
    """

    def __init__(self, x, y, sample_weight=None, dataset_path="tmp/dataset", num_processes=10):
        """
        Initializes a TFRecord_dataset instance.

        Args:
            x: Input data (can be a single array or a list of arrays).
            y: Target labels.
            sample_weight (array-like, optional): Sample weights.
            dataset_path (str, optional): Base path for storing TFRecord files (default: "tmp/dataset").
            num_processes (int, optional): Number of processes for parallel encoding.
        """
        x = x if isinstance(x, list) else [x]
        for i in range(len(x)):
            if x[i].ndim == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        if sample_weight is not None:
            if sample_weight.ndim == 1:
                sample_weight = np.expand_dims(sample_weight, axis=1)

        self.x = x
        self.y = y
        self.sample_weight = sample_weight
        self.dataset_path = dataset_path  # + str(uuid.uuid4().hex)  # create unique name for the dataset
        self.num_processes = num_processes

        # create the TFRecord dataset
        self.encode()
        self.dataset = self.decode()

    @staticmethod
    def bytes_feature(value):
        """
        Converts a value (string/byte) into a TensorFlow "bytes_list".

        Args:
            value: The value to be converted.

        Returns:
            tf.train.Feature: A TensorFlow feature containing a "bytes_list".
        """

        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def serialize_array(array):
        """
        Serializes a NumPy array using TensorFlow serialization mechanism.

        Args:
            array (np.ndarray): The NumPy array to be serialized.

        Returns:
            tf.Tensor: The serialized tensor representation.
        """
        return tf.io.serialize_tensor(array)

    ########################################################################
    # def encode(self):
    #     """
    #     encode the dataset into bytes are write to TFRecord
    #     """
    #     for i in range(0, len(self.y), self.write_buffer_size):
    #         def serialized_gen():
    #             for j in range(i, min(len(self.y), i + self.write_buffer_size)):
    #                 x_dict = {"x_{}".format(k): self.bytes_feature(
    #                     self.serialize_array(self.x[k][j])) for k in range(len(self.x))}
    #                 y_dict = {"y": self.bytes_feature(self.serialize_array(self.y[j]))}
    #                 sample_weight_dict = {} if self.sample_weight is None else \
    #                     {"sample_weight": self.bytes_feature(self.serialize_array(self.sample_weight[j]))}
    #                 feature = {**x_dict, **y_dict, **sample_weight_dict}
    #                 yield tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    #         serialized_features_dataset = tf.data.Dataset.from_generator(
    #             serialized_gen, output_types=tf.string, output_shapes=())
    #         # serialized_features_dataset = serialized_features_dataset.take(self.write_buffer_size)
    #         writer = tf.data.experimental.TFRecordWriter(self.dataset_path)
    #         writer.write(serialized_features_dataset)

    #########################################################################
    # @staticmethod
    # def write_portion(x, y, sample_weight, dataset_path, process_id, range_start, range_end):
    #     with tf.io.TFRecordWriter(dataset_path + "_part_{}.tfrecords".format(process_id)) as file_writer:
    #         for i in range(range_start, range_end):
    #             x_dict = {"x_{}".format(k): TFRecord_dataset.bytes_feature(
    #                 TFRecord_dataset.serialize_array(x[k][i])) for k in range(len(x))}
    #             y_dict = {"y": TFRecord_dataset.bytes_feature(TFRecord_dataset.serialize_array(y[i]))}
    #             sample_weight_dict = {} if sample_weight is None else \
    #                 {"sample_weight": TFRecord_dataset.bytes_feature(
    #                     TFRecord_dataset.serialize_array(sample_weight[i]))}
    #             feature = {**x_dict, **y_dict, **sample_weight_dict}
    #             record_bytes = tf.train.Example(
    #                 features=tf.train.Features(feature=feature)).SerializeToString()  # type: ignore
    #             file_writer.write(record_bytes)

    # def encode(self):
    #     """
    #     encode the dataset into bytes and write to multiple TFRecord files
    #     """
    #     range_size = len(self.y) // self.num_processes
    #     ranges = [(i * range_size, (i + 1) * range_size) for i in range(self.num_processes - 1)] + [(
    #         (self.num_processes - 1) * range_size, len(self.y))]
    #     processes = [
    #         multiprocessing.Process(
    #             target=self.write_portion,
    #             args=(self.x, self.y, self.sample_weight, self.dataset_path, i, start, end))
    #         for i, (start, end) in enumerate(ranges)]
    #     for process in processes:
    #         process.start()
    #     for process in processes:
    #         process.join()
    #     # with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
    #     #     futures = [executor.submit(self.write_portion, i, start, end) for i, (start, end) in enumerate(ranges)]
    #     #     concurrent.futures.wait(futures)

    #########################################################################
    def encode(self):
        """
        Serializes the input data, labels, and optional sample weights into TFRecord format
        and writes the data to a file.
        """
        with tf.io.TFRecordWriter(self.dataset_path + ".tfrecords") as file_writer:
            for i in range(len(self.y)):
                x_dict = {"x_{}".format(k): self.bytes_feature(
                    self.serialize_array(self.x[k][i])) for k in range(len(self.x))}
                y_dict = {"y": self.bytes_feature(self.serialize_array(self.y[i]))}
                sample_weight_dict = {} if self.sample_weight is None else \
                    {"sample_weight": self.bytes_feature(self.serialize_array(self.sample_weight[i]))}
                feature = {**x_dict, **y_dict, **sample_weight_dict}
                record_bytes = tf.train.Example(
                    features=tf.train.Features(feature=feature)).SerializeToString()  # type: ignore
                file_writer.write(record_bytes)

    def _parse_tfr_element(self, element):
        """
        Parses a single TFRecord example and restores the input data, labels, and sample weights.

        Args:
            element: A serialized TFRecord example.

        Returns:
            tuple: A tuple containing the parsed input data, labels, and (optionally) sample weights.
        """
        x_dict = {"x_{}".format(k): tf.io.FixedLenFeature([], tf.string) for k in range(len(self.x))}
        y_dict = {"y": tf.io.FixedLenFeature([], tf.string)}
        sample_weight_dict = {} if self.sample_weight is None else \
            {"sample_weight": tf.io.FixedLenFeature([], tf.string)}
        parse_dic = {**x_dict, **y_dict, **sample_weight_dict}
        example_message = tf.io.parse_single_example(element, parse_dic)

        y_i = tf.io.parse_tensor(example_message["y"], out_type=self.y.dtype)
        if len(self.x) == 1:
            x_i = tf.io.parse_tensor(example_message["x_0"], out_type=self.x[0].dtype)  # restore array from byte string
        else:
            x_i = tuple((tf.io.parse_tensor(
                    example_message["x_{}".format(k)], out_type=self.x[k].dtype) for k in range(len(self.x))))
        if self.sample_weight is None:
            return x_i, y_i
        else:
            sw_i = tf.io.parse_tensor(example_message["sample_weight"], out_type=self.sample_weight.dtype)
            return x_i, y_i, sw_i

    def decode(self):
        """
        Loads the TFRecord dataset and parses the examples.

        Returns:
            tf.data.Dataset: A TensorFlow dataset object.
        """
        dataset = tf.data.TFRecordDataset(self.dataset_path + ".tfrecords").map(self._parse_tfr_element)
        return dataset

    # def decode(self):
    #     """
    #     load TFRecord and parse the elements back to dataset
    #     """
    #     return tf.data.TFRecordDataset(
    #         [self.dataset_path + "_part_{}.tfrecords".format(file_id) for file_id in range(self.num_processes)]).map(
    #             self._parse_tfr_element)
