from OT_deep_score_src.general_utilities import Model_task, MAX_DISTANCE
from OT_deep_score_src.models_inter import NN_model
from OT_deep_score_src.models_inter import validate_model_task, get_optimizer

import tensorflow as tf


class NN_model_with_pretrained(NN_model):
    @property
    def pretrained_model(self):
        return self._pretrained_model

    @pretrained_model.setter
    def pretrained_model(self, pretrained_model):
        self._pretrained_model = pretrained_model.model if \
            isinstance(pretrained_model, NN_model) else pretrained_model

    def save_model_instance(self, file_path_and_name):
        if self.pretrained_model is not None:
            # save pretrained model
            self.pretrained_model.save(file_path_and_name + "_pretrained.h5")
            # We do not want to save the pretrianed model as a pickle file
            pretrained_model = self.pretrained_model
            self.pretrained_model = None
            super(NN_model_with_pretrained, self).save_model_instance(file_path_and_name)
            self.pretrained_model = pretrained_model
        else:
            super(NN_model_with_pretrained, self).save_model_instance(file_path_and_name)

    @classmethod
    def load_model_instance(cls, file_path_and_name):
        model_instance = super(NN_model_with_pretrained).load_model_instance(file_path_and_name)
        model_instance.pretrained_model = tf.keras.models.load_model(  # type: ignore
            file_path_and_name + "_pretrained.h5")

        return model_instance


class Distance_model(NN_model):
    def __init__(self, batch_size, epochs, model_task, learning_rate, optimizer,
                 one_hot_to_categorical):
        validate_model_task(model_task)
        super(Distance_model, self).__init__(batch_size, epochs)
        self._model_task = model_task
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._one_hot_to_categorical = one_hot_to_categorical

    def compile(self):
        if self.model is not None:
            optimizer = get_optimizer(self._optimizer, self._learning_rate)
            if self._model_task == Model_task.CLASSIFICATION_TASK:
                self.model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy,
                                   weighted_metrics=[])
            else:
                self.model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error,
                                   weighted_metrics=[])
            print(self.model.summary())
        else:
            raise ValueError("Cannot complie None model")

    def top_layer(self, inputs, pre_top_output):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            outputs = tf.keras.layers.Dense(MAX_DISTANCE + 1, activation=tf.keras.activations.softmax)(pre_top_output)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        else:
            outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)(pre_top_output)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def fit(self, x, y, validation_data=None, sample_weight=None, val_size=None, verbose=2, choose_best_epoch=False,
            train_fit_log_path=None, early_stopping=True, stratify_in_split=False, use_tfrecords=False):
        if self._one_hot_to_categorical:
            x, validation_data = NN_model.one_hot_to_categorical(x, validation_data)
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            # convert labels into categorical
            y = tf.keras.utils.to_categorical(y, MAX_DISTANCE+1)
            if validation_data is not None:
                validation_data = list(validation_data)
                validation_data[1] = tf.keras.utils.to_categorical(validation_data[1], MAX_DISTANCE+1)
                validation_data = tuple(validation_data)

        super(Distance_model, self).fit(
            x=x, y=y, validation_data=validation_data, sample_weight=sample_weight,
            val_size=val_size, verbose=verbose, choose_best_epoch=choose_best_epoch,
            train_fit_log_path=train_fit_log_path, early_stopping=early_stopping,
            stratify_in_split=stratify_in_split, use_tfrecords=use_tfrecords)

    def predict(self, x):
        if self._one_hot_to_categorical:
            x, _ = NN_model.one_hot_to_categorical(x)
        return super(Distance_model, self).predict(x)


class Cleavage_model(NN_model):
    def __init__(self, batch_size, epochs, model_task, learning_rate, optimizer,
                 one_hot_to_categorical):
        validate_model_task(model_task)
        super(Cleavage_model, self).__init__(batch_size, epochs)
        self._model_task = model_task
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._one_hot_to_categorical = one_hot_to_categorical

    def compile(self):
        if self.model is not None:
            optimizer = get_optimizer(self._optimizer, self._learning_rate)
            if self._model_task == Model_task.CLASSIFICATION_TASK:
                self.model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy,
                                   metrics=[tf.keras.metrics.AUC(curve="PR", name="AUPR"),
                                            tf.keras.metrics.AUC(curve="ROC", name="AUC")],
                                   weighted_metrics=[])
            else:
                self.model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error,
                                   weighted_metrics=[])
            print(self.model.summary())
        else:
            raise ValueError("Cannot complie None model")

    def top_layer(self, inputs, pre_top_output):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            outputs = tf.keras.layers.Dense(
                1, activation=tf.keras.activations.sigmoid, name="tf_dense")(pre_top_output)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        else:
            outputs = tf.keras.layers.Dense(
                1, activation=tf.keras.activations.linear, name="tf_dense")(pre_top_output)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def fit(self, x, y, validation_data=None, sample_weight=None, val_size=None, verbose=2, choose_best_epoch=False,
            train_fit_log_path=None, early_stopping=True, stratify_in_split=False, use_tfrecords=False):
        if self._one_hot_to_categorical:
            x, validation_data = NN_model.one_hot_to_categorical(x, validation_data)

        super(Cleavage_model, self).fit(
            x=x, y=y, validation_data=validation_data, sample_weight=sample_weight,
            val_size=val_size, verbose=verbose, choose_best_epoch=choose_best_epoch,
            train_fit_log_path=train_fit_log_path, early_stopping=early_stopping,
            stratify_in_split=stratify_in_split, use_tfrecords=use_tfrecords)

    def predict(self, x):
        if self._one_hot_to_categorical:
            x, _ = NN_model.one_hot_to_categorical(x)
        return super(Cleavage_model, self).predict(x)


class D_1(Distance_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), embed_dim=44, embed_dropout=0.2,
                 dense_layers=(128, 64), activation_funs=("relu", "relu"), optimizer='adam'):
        super(D_1, self).__init__(batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=True)
        self._input_length = input_shape[0]
        self._input_dim = input_shape[1]
        self._embed_dim = embed_dim
        self._embed_dropout = embed_dropout
        self._dense_layers = dense_layers
        self._activation_funs = activation_funs

    def construct(self):
        inputs = tf.keras.layers.Input(shape=(self._input_length))

        embedding_layer = tf.keras.layers.Embedding(self._input_dim, self._embed_dim,
                                                    input_length=self._input_length)
        x = embedding_layer(inputs)
        x = tf.keras.layers.Flatten()(x)
        for i in range(len(self._dense_layers)):
            x = tf.keras.layers.Dense(self._dense_layers[i],
                                      activation=self._activation_funs[i])(x)

        self.top_layer(inputs=inputs, pre_top_output=x)
        self.compile()


class D_2(Distance_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), embed_dim=44, embed_dropout=0.2,
                 dense_layers=(128, 64), activation_funs=("relu", "relu"), optimizer='adam'):
        super(D_2, self).__init__(batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=True)
        self._input_length = input_shape[0]
        self._input_dim = input_shape[1]
        self._embed_dim = embed_dim
        self._embed_dropout = embed_dropout
        self._dense_layers = dense_layers
        self._activation_funs = activation_funs

    def construct(self):
        inputs = tf.keras.layers.Input(shape=(self._input_length))

        embedding_layer = tf.keras.layers.Embedding(self._input_dim, self._embed_dim,
                                                    input_length=self._input_length)
        x = embedding_layer(inputs)
        gru = tf.keras.layers.GRU(64, return_sequences=True)
        x = gru(x)
        x = tf.keras.layers.Flatten()(x)
        for i in range(len(self._dense_layers)):
            x = tf.keras.layers.Dense(self._dense_layers[i],
                                      activation=self._activation_funs[i])(x)

        self.top_layer(inputs=inputs, pre_top_output=x)
        self.compile()


class D_3(Distance_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), dense_layers=(128, 64), activation_funs=("relu", "relu"), optimizer='adam'):
        super(D_3, self).__init__(
            batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=False)
        self._input_length = input_shape[0]
        self._input_dim = input_shape[1]
        self._dense_layers = dense_layers
        self._activation_funs = activation_funs

    def construct(self):
        inputs = tf.keras.layers.Input(shape=(self._input_length, self._input_dim))

        gru = tf.keras.layers.GRU(64, return_sequences=True)
        x = gru(inputs)
        x = tf.keras.layers.Flatten()(x)
        for i in range(len(self._dense_layers)):
            x = tf.keras.layers.Dense(self._dense_layers[i],
                                      activation=self._activation_funs[i])(x)

        self.top_layer(inputs=inputs, pre_top_output=x)
        self.compile()


class D2C_2(Cleavage_model, NN_model_with_pretrained):
    """
    Was Cleavage_model_2
    """
    def __init__(self, model_task, pretrained_model, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), optimizer='adam'):
        super().__init__(batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=True)
        self.pretrained_model = pretrained_model
        self._additional_input_size = None
        if isinstance(input_shape, list):
            self._additional_input_size = input_shape[1]

    def construct(self):
        base_model = self.pretrained_model
        if base_model is None:
            raise ValueError("When base_model is None, construct is not allowed")
        if self._additional_input_size is not None:
            inputs_2 = tf.keras.layers.Input(shape=(self._additional_input_size), name="tf_additional_input")
            inputs = [base_model.input, inputs_2]
        else:
            inputs_2 = None
            inputs = base_model.input

        pre_top_output = base_model.layers[-2].output
        if inputs_2 is not None:
            pre_top_output = tf.keras.layers.Concatenate(name="tf_concatenate")([pre_top_output, inputs_2])
        self.top_layer(inputs=inputs, pre_top_output=pre_top_output)
        self.compile()


class D2C_2_1(Cleavage_model, NN_model_with_pretrained):
    def __init__(self, model_task, pretrained_model, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), dense_layers=(128, 64), optimizer='adam'):
        super().__init__(batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=True)
        self.pretrained_model = pretrained_model
        self._additional_input_size = None
        if isinstance(input_shape, list):
            self._additional_input_size = input_shape[1]
        self._dense_layers = dense_layers

    def construct(self):
        base_model = self.pretrained_model
        if base_model is None:
            raise ValueError("When base_model is None, construct is not allowed")
        for layer in base_model.layers[:-(1+len(self._dense_layers))]:
            layer.trainable = False

        if self._additional_input_size is not None:
            inputs_2 = tf.keras.layers.Input(shape=(self._additional_input_size), name="tf_additional_input")
            inputs = [base_model.input, inputs_2]
        else:
            inputs_2 = None
            inputs = base_model.input

        pre_top_output = base_model.layers[-2].output
        if inputs_2 is not None:
            pre_top_output = tf.keras.layers.Concatenate(name="tf_concatenate")([pre_top_output, inputs_2])
        self.top_layer(inputs=inputs, pre_top_output=pre_top_output)
        self.compile()


class D2C_3(Cleavage_model, NN_model_with_pretrained):
    def __init__(self, model_task, pretrained_model, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), optimizer='adam'):
        super().__init__(batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=False)
        self.pretrained_model = pretrained_model
        self._additional_input_size = None
        if isinstance(input_shape, list):
            self._additional_input_size = input_shape[1]

    def construct(self):
        base_model = self.pretrained_model
        if base_model is None:
            raise ValueError("When base_model is None, construct is not allowed")
        if self._additional_input_size is not None:
            inputs_2 = tf.keras.layers.Input(shape=(self._additional_input_size), name="tf_additional_input")
            inputs = [base_model.input, inputs_2]
        else:
            inputs_2 = None
            inputs = base_model.input

        pre_top_output = base_model.layers[-2].output
        if inputs_2 is not None:
            pre_top_output = tf.keras.layers.Concatenate(name="tf_concatenate")([pre_top_output, inputs_2])
        self.top_layer(inputs=inputs, pre_top_output=pre_top_output)
        self.compile()


class C_1(Cleavage_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), embed_dim=44, embed_dropout=0.2,
                 dense_layers=(128, 64), activation_funs=("relu", "relu"), optimizer='adam'):
        super(C_1, self).__init__(batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=True)
        self._additional_input_size = None
        if isinstance(input_shape, list):
            self._additional_input_size = input_shape[1]
            input_shape = input_shape[0]
        self._input_length = input_shape[0]
        self._input_dim = input_shape[1]
        self._embed_dim = embed_dim
        self._embed_dropout = embed_dropout
        self._dense_layers = dense_layers
        self._activation_funs = activation_funs

    def construct(self):
        inputs_1 = tf.keras.layers.Input(shape=(self._input_length))
        if self._additional_input_size is not None:
            inputs_2 = tf.keras.layers.Input(shape=(self._additional_input_size))
            inputs = [inputs_1, inputs_2]
        else:
            inputs_2 = None
            inputs = inputs_1

        embedding_layer = tf.keras.layers.Embedding(self._input_dim, self._embed_dim,
                                                    input_length=self._input_length)
        x = embedding_layer(inputs_1)
        x = tf.keras.layers.Flatten()(x)
        if inputs_2 is not None:
            x = tf.keras.layers.Concatenate()([x, inputs_2])
        for i in range(len(self._dense_layers)):
            x = tf.keras.layers.Dense(self._dense_layers[i],
                                      activation=self._activation_funs[i])(x)

        self.top_layer(inputs=inputs, pre_top_output=x)
        self.compile()


class C_2(Cleavage_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), embed_dim=44, embed_dropout=0.2,
                 dense_layers=(128, 64), activation_funs=("relu", "relu"), optimizer='adam'):
        super(C_2, self).__init__(batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=True)
        self._additional_input_size = None
        if isinstance(input_shape, list):
            self._additional_input_size = input_shape[1]
            input_shape = input_shape[0]
        self._input_length = input_shape[0]
        self._input_dim = input_shape[1]
        self._embed_dim = embed_dim
        self._embed_dropout = embed_dropout
        self._dense_layers = dense_layers
        self._activation_funs = activation_funs

    def construct(self):
        inputs_1 = tf.keras.layers.Input(shape=(self._input_length))
        if self._additional_input_size is not None:
            inputs_2 = tf.keras.layers.Input(shape=(self._additional_input_size))
            inputs = [inputs_1, inputs_2]
        else:
            inputs_2 = None
            inputs = inputs_1

        embedding_layer = tf.keras.layers.Embedding(self._input_dim, self._embed_dim,
                                                    input_length=self._input_length)
        x = embedding_layer(inputs_1)
        gru = tf.keras.layers.GRU(64, return_sequences=True)
        x = gru(x)
        x = tf.keras.layers.Flatten()(x)
        if inputs_2 is not None:
            x = tf.keras.layers.Concatenate()([x, inputs_2])
        for i in range(len(self._dense_layers)):
            x = tf.keras.layers.Dense(self._dense_layers[i],
                                      activation=self._activation_funs[i])(x)

        self.top_layer(inputs=inputs, pre_top_output=x)
        self.compile()


class C_3(Cleavage_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.001,
                 input_shape=(24, 25), dense_layers=(128, 64), activation_funs=("relu", "relu"), optimizer='adam'):
        super(C_3, self).__init__(
            batch_size, epochs, model_task, learning_rate, optimizer, one_hot_to_categorical=False)
        self._additional_input_size = None
        if isinstance(input_shape, list):
            self._additional_input_size = input_shape[1]
            input_shape = input_shape[0]
        self._input_length = input_shape[0]
        self._input_dim = input_shape[1]
        self._dense_layers = dense_layers
        self._activation_funs = activation_funs

    def construct(self):
        inputs_1 = tf.keras.layers.Input(shape=(self._input_length, self._input_dim))
        if self._additional_input_size is not None:
            inputs_2 = tf.keras.layers.Input(shape=(self._additional_input_size))
            inputs = [inputs_1, inputs_2]
        else:
            inputs_2 = None
            inputs = inputs_1

        gru = tf.keras.layers.GRU(64, return_sequences=True)
        x = gru(inputs_1)
        x = tf.keras.layers.Flatten()(x)
        if inputs_2 is not None:
            x = tf.keras.layers.Concatenate()([x, inputs_2])
        for i in range(len(self._dense_layers)):
            x = tf.keras.layers.Dense(self._dense_layers[i],
                                      activation=self._activation_funs[i])(x)

        self.top_layer(inputs=inputs, pre_top_output=x)
        self.compile()
