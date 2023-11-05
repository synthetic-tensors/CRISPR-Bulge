from abc import ABC, abstractmethod

import pickle
import shutil
import uuid

from OT_deep_score_src.general_utilities import Model_task, SEED, FILES_DIR

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


def validate_model_task(model_task):
    if model_task not in [Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK]:
        raise ValueError("Model task should be one of {}".format(
            [Model_task.CLASSIFICATION_TASK, Model_task.REGRESSION_TASK]))


def split_to_train_and_val(x, y, sample_weight, val_size, validation_data, stratify_in_split):
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
    def __init__(self):
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model.model if isinstance(model, Model) else model

    @abstractmethod
    def construct(self):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, file_path_and_name):
        pass

    @abstractmethod
    def load(self, file_path_and_name):
        pass

    def save_model_instance(self, file_path_and_name):
        self.save(file_path_and_name)
        # We do not want to save the model itself as a pickle file
        model = self.model
        self.model = None
        with open(file_path_and_name + ".pkl", 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.model = model

    @classmethod
    def load_model_instance(cls, file_path_and_name):
        with open(file_path_and_name + ".pkl", 'rb') as f:
            model_instance: Model = pickle.load(f)
        model_instance.load(file_path_and_name)

        return model_instance


class Sklearn_model(Model):
    def __init__(self, model_task):
        validate_model_task(model_task)
        self._model_task = model_task
        super().__init__()

    def fit(self, x, y, sample_weight=None):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, x):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            return self.model.predict_proba(x)  # [:, 1]
        else:
            return self.model.predict(x)

    def save(self, file_path_and_name, suffix=".skl"):
        with open(file_path_and_name + suffix, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path_and_name, suffix=".skl"):
        with open(file_path_and_name + suffix, 'rb') as f:
            self.model = pickle.load(f)


def get_optimizer(optimizer_name, learning_rate):
    optimizer = tf.keras.optimizers.get(
            {"class_name": optimizer_name, "config": {"learning_rate": learning_rate}})

    return optimizer


class NN_model(Model):
    PREDICT_BATCH_SIZE = 4096

    def __init__(self, batch_size, epochs):
        # clear the session to clear old layer names
        tf.keras.backend.clear_session()
        self._batch_size = batch_size
        self._epochs = epochs
        super().__init__()

    @abstractmethod
    def compile(self):
        pass

    @staticmethod
    def one_hot_to_categorical(x, validation_data=None):
        # convert one-hot into categorical
        # If there are multiple inputs, then the first input is the one-hot
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
    def model_checkpoint_callbak(checkpoint_filepath, monitor='val_loss', mode='min'):
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
            train_fit_log_path, early_stopping, stratify_in_split, use_tfrecords):
        """
        choose_best_epoch is set to false at default since I prefere early stopping
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        x, y, sample_weight, validation_data = split_to_train_and_val(
            x, y, sample_weight, val_size, validation_data, stratify_in_split)
        callbacks = []
        # define checkpoint callback if needed
        checkpoint_filepath = FILES_DIR + "tmp" + str(uuid.uuid4().hex)  # create unique tmp folder name
        if early_stopping and validation_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True))
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
                       callbacks=callbacks)

        if choose_best_epoch and validation_data is not None:
            self.model.load_weights(checkpoint_filepath + "/checkpoint")
            shutil.rmtree(checkpoint_filepath, ignore_errors=True)

        # this should force clear after fit
        tf.keras.backend.clear_session()

    def predict(self, x):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        # TODO: make it flexible to other batch sizes
        return self.model.predict(x, batch_size=NN_model.PREDICT_BATCH_SIZE).squeeze()

    def save(self, file_path_and_name):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.save(file_path_and_name + ".h5")

    def load(self, file_path_and_name):
        self.model = tf.keras.models.load_model(file_path_and_name + ".h5")


class TFRecord_dataset():
    def __init__(self, x, y, sample_weight=None, dataset_path="tmp/dataset", num_processes=10):
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
        Returns a bytes_list from a string / byte.
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def serialize_array(array):
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
        encode the dataset into bytes are write to TFRecord
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
        x_dict = {"x_{}".format(k): tf.io.FixedLenFeature([], tf.string) for k in range(len(self.x))}
        y_dict = {"y": tf.io.FixedLenFeature([], tf.string)}
        sample_weight_dict = {} if self.sample_weight is None else \
            {"sample_weight": tf.io.FixedLenFeature([], tf.string)}
        parse_dic = {**x_dict, **y_dict, **sample_weight_dict}
        example_message = tf.io.parse_single_example(element, parse_dic)

        y_i = tf.io.parse_tensor(example_message['y'], out_type=self.y.dtype)
        if len(self.x) == 1:
            x_i = tf.io.parse_tensor(example_message['x_0'], out_type=self.x[0].dtype)  # restore array from byte string
        else:
            x_i = tuple((tf.io.parse_tensor(
                    example_message["x_{}".format(k)], out_type=self.x[k].dtype) for k in range(len(self.x))))
        if self.sample_weight is None:
            return x_i, y_i
        else:
            sw_i = tf.io.parse_tensor(example_message['sample_weight'], out_type=self.sample_weight.dtype)
            return x_i, y_i, sw_i

    def decode(self):
        """
        load TFRecord and parse the elements back to dataset
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
