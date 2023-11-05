from OT_deep_score_src.general_utilities import Model_task, Xgboost_tf_type
from OT_deep_score_src.models_inter import Model, Sklearn_model, NN_model
from OT_deep_score_src.models_inter import validate_model_task, get_optimizer

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV

import tensorflow as tf


# TODO: This class (and catboost), like some of the NN model, has pretrained model,
# I'm not providing it as argument to the fit is the right thing
class XGboost_model(Model):
    @staticmethod
    def validate_transfer_learning_type(transfer_learning_type):
        if transfer_learning_type not in [Xgboost_tf_type.UPDATE, Xgboost_tf_type.ADD, Xgboost_tf_type.NONE]:
            raise ValueError("Transfer learning type should be one of {}".format(
                [Xgboost_tf_type.UPDATE, Xgboost_tf_type.ADD, Xgboost_tf_type.NONE]))

    def __init__(self, model_task, max_depth=10, learning_rate=0.1, n_estimators=1000,
                 transfer_learning_type=None, gpu=True, nthread=50):
        validate_model_task(model_task)
        XGboost_model.validate_transfer_learning_type(transfer_learning_type)
        self._model_task = model_task
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._transfer_learning_type = transfer_learning_type
        self._gpu = gpu
        self._nthread = nthread
        super().__init__()

    def construct(self):
        # set transfer-learning setting if needed
        transfer_learning_args = {} if self._gpu else {}
        if self._transfer_learning_type == Xgboost_tf_type.UPDATE:
            # does not work with GPU
            transfer_learning_args = {'process_type': 'update', 'updater': 'refresh'}
        else:
            transfer_learning_args = {'tree_method': 'gpu_hist'} if self._gpu else {}

        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = XGBClassifier(max_depth=self._max_depth,
                                       learning_rate=self._learning_rate,
                                       n_estimators=self._n_estimators,
                                       nthread=self._nthread,
                                       **transfer_learning_args)
        else:
            self.model = XGBRegressor(max_depth=self._max_depth,
                                      learning_rate=self._learning_rate,
                                      n_estimators=self._n_estimators,
                                      nthread=self._nthread,
                                      **transfer_learning_args)

    def fit(self, x, y, sample_weight=None, pretrained_model=None):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.fit(x, y, sample_weight=sample_weight, xgb_model=pretrained_model)

    def predict(self, x):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            assert isinstance(self._model, XGBClassifier)
            return self.model.predict_proba(x)  # [:, 1]
        else:
            assert isinstance(self._model, XGBRegressor)
            return self.model.predict(x)

    def save(self, file_path_and_name):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.save_model(file_path_and_name + ".xgb")

    def load(self, file_path_and_name):
        self.construct()
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.load_model(file_path_and_name + ".xgb")


class CatBoost_model(Model):
    def __init__(self, model_task, max_depth=10, learning_rate=0.1, n_estimators=1000, gpu=True):
        validate_model_task(model_task)
        self._model_task = model_task
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._gpu = gpu
        super().__init__()

    def construct(self):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = CatBoostClassifier(max_depth=self._max_depth,
                                            learning_rate=self._learning_rate,
                                            n_estimators=self._n_estimators,
                                            task_type="GPU" if self._gpu else "CPU")
        else:
            self.model = CatBoostRegressor(max_depth=self._max_depth,
                                           learning_rate=self._learning_rate,
                                           n_estimators=self._n_estimators,
                                           task_type="GPU" if self._gpu else "CPU")

    def fit(self, x, y, sample_weight=None):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, x):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            assert isinstance(self._model, CatBoostClassifier)
            return self.model.predict_proba(x)  # [:, 1]
        else:
            assert isinstance(self._model, CatBoostRegressor)
            return self.model.predict(x)

    def save(self, file_path_and_name):
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.save_model(file_path_and_name + ".catb")

    def load(self, file_path_and_name):
        self.construct()
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.load_model(file_path_and_name + ".catb")


class SVM_model(Sklearn_model):
    def construct(self):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = svm.SVC()
        else:
            self.model = svm.SVR()

    def save(self, file_path_and_name):
        super().save(file_path_and_name, ".svm")

    def load(self, file_path_and_name):
        super().load(file_path_and_name, ".svm")


class SVM_linear_model(Sklearn_model):
    def construct(self):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = svm.LinearSVC()
            self.model = CalibratedClassifierCV(self._model)
        else:
            self.model = svm.LinearSVR()

    def save(self, file_path_and_name):
        super().save(file_path_and_name, ".svml")

    def load(self, file_path_and_name):
        super().load(file_path_and_name, ".svml")


class SGD_model(Sklearn_model):
    def construct(self):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = linear_model.SGDClassifier()
            self.model = CalibratedClassifierCV(self._model)
        else:
            self.model = linear_model.SGDRegressor()

    def save(self, file_path_and_name):
        super().save(file_path_and_name, ".sgd")

    def load(self, file_path_and_name):
        super().load(file_path_and_name, ".sgd")


class AdaBoost_model(Sklearn_model):
    def __init__(self, model_task, n_estimators=1000, learning_rate=0.1):
        super().__init__(model_task)
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate

    def construct(self):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = AdaBoostClassifier(learning_rate=self._learning_rate,
                                            n_estimators=self._n_estimators)
        else:
            self.model = AdaBoostRegressor(learning_rate=self._learning_rate,
                                           n_estimators=self._n_estimators)

    def save(self, file_path_and_name):
        super().save(file_path_and_name, ".adab")

    def load(self, file_path_and_name):
        super().load(file_path_and_name, ".adab")


class RandomForest_model(Sklearn_model):
    def __init__(self, model_task, n_estimators=1000, max_depth=None, n_jobs=50):
        super().__init__(model_task)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._n_jobs = n_jobs

    def construct(self):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = RandomForestClassifier(max_depth=self._max_depth,
                                                n_estimators=self._n_estimators,
                                                n_jobs=self._n_jobs)
        else:
            self.model = RandomForestRegressor(max_depth=self._max_depth,
                                               n_estimators=self._n_estimators,
                                               n_jobs=self._n_jobs)

    def save(self, file_path_and_name):
        super().save(file_path_and_name, ".rf")

    def load(self, file_path_and_name):
        super().load(file_path_and_name, ".rf")


class Lasso_model(Sklearn_model):
    def __init__(self, model_task, max_iter=1000, alpha=1, reg_penalty="l1", reg_solver="saga"):
        super().__init__(model_task)
        self._max_iter = max_iter
        self._reg_penalty = reg_penalty
        self._reg_solver = reg_solver
        self._alpha = alpha

    def construct(self):
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = linear_model.Lasso(max_iter=self._max_iter, alpha=self._alpha)
        else:
            # Note: I using here the Logistic Regression model with L1 penalty
            # 'lbfgs'- the default - ['l2', 'none'], 'saga' - ['elasticnet', 'l1', 'l2', 'none']
            # for small datasets, ‘liblinear’ is a good choice,
            # whereas ‘sag’ and ‘saga’ are faster for large ones;
            self.model = linear_model.LogisticRegression(
                max_iter=self._max_iter, penalty=self._reg_penalty, solver=self._reg_solver)  # type: ignore

    def save(self, file_path_and_name):
        super().save(file_path_and_name, ".lasso")

    def load(self, file_path_and_name):
        super().load(file_path_and_name, ".lasso")


class MLP_model(NN_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.1,
                 dense_layers=(128,), activation_funs=("relu",), optimizer='adam',
                 input_shape=(600, )):
        validate_model_task(model_task)
        self._model_task = model_task
        self._learning_rate = learning_rate
        self._dense_layers = dense_layers
        self._activation_funs = activation_funs
        self._optimizer = optimizer
        self._input_shape = input_shape
        super().__init__(batch_size, epochs)

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

    def construct(self):
        inputs = tf.keras.layers.Input(shape=self._input_shape)
        x = inputs
        for i in range(len(self._dense_layers)):
            x = tf.keras.layers.Dense(self._dense_layers[i],
                                      activation=self._activation_funs[i])(x)
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(x)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        else:
            outputs = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)(x)
            self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.compile()
