"""
This module contains the base models creation
"""
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
# I am not providing it as argument to the fit is the right thing
class XGboost_model(Model):
    """
    A class representing an XGBoost model, providing flexible configuration
    and support for transfer learning.
    """

    @staticmethod
    def validate_transfer_learning_type(transfer_learning_type):
        """
        Validates the provided transfer learning type.

        Args:
            transfer_learning_type: The type of transfer learning to use (e.g., "UPDATE", "ADD", or "NONE").

        Raises:
            ValueError: If the transfer_learning_type is invalid.
        """
        if transfer_learning_type not in [Xgboost_tf_type.UPDATE, Xgboost_tf_type.ADD, Xgboost_tf_type.NONE]:
            raise ValueError("Transfer learning type should be one of {}".format(
                [Xgboost_tf_type.UPDATE, Xgboost_tf_type.ADD, Xgboost_tf_type.NONE]))

    def __init__(self, model_task, max_depth=10, learning_rate=0.1, n_estimators=1000,
                 transfer_learning_type=None, gpu=True, nthread=50):
        """
        Initializes an XGboost_model instance.

        Args:
            model_task: The type of task ("CLASSIFICATION_TASK" or "REGRESSION_TASK").
            max_depth: Maximum depth of XGBoost trees (default: 10).
            learning_rate: Learning rate for the XGBoost model (default: 0.1).
            n_estimators: Number of trees in the ensemble (default: 1000).
            transfer_learning_type: Type of transfer learning ("UPDATE", "ADD", or "NONE", default: None).
            gpu: If True, use GPU acceleration for training (default: True).
            nthread: Number of CPU threads to use (default: 50).
        """
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
        """
        Constructs the XGBoost model with appropriate settings, including
        configuration for transfer learning and GPU acceleration.
        """
        # set transfer-learning setting if needed
        transfer_learning_args = {} if self._gpu else {}
        if self._transfer_learning_type == Xgboost_tf_type.UPDATE:
            # does not work with GPU
            transfer_learning_args = {"process_type": "update", "updater": "refresh"}
        else:
            transfer_learning_args = {"tree_method": "gpu_hist"} if self._gpu else {}

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
        """
        Fits the XGBoost model to the provided data.

        Args:
            x: Training data (features).
            y: Target data (labels).
            sample_weight: Optional weights for each sample.
            pretrained_model: Optional, an existing XGBoost model for transfer learning.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.fit(x, y, sample_weight=sample_weight, xgb_model=pretrained_model)

    def predict(self, x):
        """
        Generates predictions using the trained XGBoost model.

        Args:
            x: Input data for prediction.

        Returns:
            Predictions from the model. For classification tasks, returns
            class probabilities. For regression tasks, returns predicted values.

        Raises:
            TypeError: If the model has not been trained.
            AssertionError: If the model type is mismatched with the task type.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            assert isinstance(self._model, XGBClassifier)
            return self.model.predict_proba(x)  # [:, 1]
        else:
            assert isinstance(self._model, XGBRegressor)
            return self.model.predict(x)

    def save(self, file_path_and_name):
        """
        Saves the trained XGBoost model to a file.

        Args:
            file_path_and_name: The path and filename where the model should be saved.
                                (The ".xgb" extension will be added automatically.)
        Raises:
            TypeError: If the model has not been trained.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.save_model(file_path_and_name + ".xgb")

    def load(self, file_path_and_name):
        """
        Loads a previously saved XGBoost model from a file.

        Args:
            file_path_and_name: The path and filename of the saved model
                                (".xgb" extension will be added automatically.)
        Raises:
            TypeError: If the model has not been initialized.
        """
        self.construct()
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.load_model(file_path_and_name + ".xgb")


class CatBoost_model(Model):
    """
    A class representing a CatBoost model, providing flexible configuration and support for GPU acceleration.
    """

    def __init__(self, model_task, max_depth=10, learning_rate=0.1, n_estimators=1000, gpu=True):
        """
        Initializes a CatBoost_model instance.

        Args:
            model_task: The type of task ("CLASSIFICATION_TASK" or "REGRESSION_TASK").
            max_depth: Maximum depth of CatBoost trees (default: 10).
            learning_rate: Learning rate for the CatBoost model (default: 0.1).
            n_estimators: Number of trees in the ensemble (default: 1000).
            gpu: If True, use GPU acceleration for training (default: True).
        """
        validate_model_task(model_task)
        self._model_task = model_task
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._gpu = gpu
        super().__init__()

    def construct(self):
        """
        Constructs the CatBoost model with appropriate settings, including
        configuration for GPU acceleration.
        """
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
        """
        Fits the CatBoost model to the provided data.

        Args:
            x: Training data (features).
            y: Target data (labels).
            sample_weight: Optional weights for each sample.
        Raises:
            TypeError: If the model has not been trained.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, x):
        """
        Generates predictions using the trained CatBoost model.

        Args:
            x: Input data for prediction.

        Returns:
            Predictions from the model. For classification tasks, returns
            class probabilities. For regression tasks, returns predicted values.

        Raises:
            TypeError: If the model has not been trained.
            AssertionError: If the model type is mismatched with the task type.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            assert isinstance(self._model, CatBoostClassifier)
            return self.model.predict_proba(x)  # [:, 1]
        else:
            assert isinstance(self._model, CatBoostRegressor)
            return self.model.predict(x)

    def save(self, file_path_and_name):
        """
        Saves the trained CatBoost model to a file.

        Args:
            file_path_and_name: The path and filename where the model should be saved.
                                (The ".catb" extension will be added automatically.)
        Raises:
            TypeError: If the model has not been trained.
        """
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.save_model(file_path_and_name + ".catb")

    def load(self, file_path_and_name):
        """
        Loads a previously saved CatBoost model from a file.

        Args:
            file_path_and_name: The path and filename of the saved model
                                (The ".catb" extension will be added automatically.)
        Raises:
            TypeError: If the model has not been initialized.
        """
        self.construct()
        if self.model is None:
            raise TypeError("The model attribute shuld be a model and not None")
        self.model.load_model(file_path_and_name + ".catb")


class SVM_model(Sklearn_model):
    """
    A class representing a Support Vector Machine (SVM) model using scikit-learn.
    """

    def construct(self):
        """
        Constructs the SVM model. Selects between "SVC" for classification and "SVR" for regression tasks.
        """
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = svm.SVC()
        else:
            self.model = svm.SVR()

    def save(self, file_path_and_name):
        """
        Saves the trained SVM model to a file.

        Args:
            file_path_and_name: The path and filename where the model should be saved.
                                (The ".svm" extension will be added automatically.)
        """
        super().save(file_path_and_name, ".svm")

    def load(self, file_path_and_name):
        """
        Loads a previously saved SVM model from a file.

        Args:
            file_path_and_name: The path and filename of the saved model
                                (The ".svm" extension will be added automatically.)
        """
        super().load(file_path_and_name, ".svm")


class SVM_linear_model(Sklearn_model):
    """
    A class representing a Support Vector Machine model with a linear kernel,
    using scikit-learn.  Includes calibration for classification tasks.
    """

    def construct(self):
        """
        Constructs the linear SVM model. Uses "LinearSVC" for classification,
         and "LinearSVR" for regression. Applies calibration for classification.
        """
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = svm.LinearSVC()
            self.model = CalibratedClassifierCV(self._model)
        else:
            self.model = svm.LinearSVR()

    def save(self, file_path_and_name):
        """
        Saves the trained linear SVM model to a file.

        Args:
            file_path_and_name: The path and filename where the model should be saved.
                                (The ".svml" extension will be added automatically.)
        """
        super().save(file_path_and_name, ".svml")

    def load(self, file_path_and_name):
        """
        Loads a previously saved linear SVM model from a file.

        Args:
            file_path_and_name: The path and filename of the saved model
                                (The ".svml" extension will be added automatically.)
        """
        super().load(file_path_and_name, ".svml")


class SGD_model(Sklearn_model):
    """
    A class representing a model using Stochastic Gradient Descent (SGD)
    from scikit-learn. Includes calibration for classification tasks.
    """

    def construct(self):
        """
        Constructs the SGD model. Uses "SGDClassifier" for classification,
         and "SGDRegressor" for regression. Applies calibration for classification.
        """
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = linear_model.SGDClassifier()
            self.model = CalibratedClassifierCV(self._model)
        else:
            self.model = linear_model.SGDRegressor()

    def save(self, file_path_and_name):
        """
        Saves the trained SGD model to a file.

        Args:
            file_path_and_name: The path and filename where the model should be saved.
                                (The ".sgd" extension will be added automatically.)
        """
        super().save(file_path_and_name, ".sgd")

    def load(self, file_path_and_name):
        """
        Loads a previously saved SGD model from a file.

        Args:
            file_path_and_name: The path and filename of the saved model
                                (The ".sgd" extension will be added automatically.)
        """
        super().load(file_path_and_name, ".sgd")


class AdaBoost_model(Sklearn_model):
    """
    A class representing an AdaBoost ensemble model using scikit-learn.
    AdaBoost combines multiple weak learners to create a stronger model.
    """

    def __init__(self, model_task, n_estimators=1000, learning_rate=0.1):
        """
        Initializes an AdaBoost_model instance.

        Args:
            model_task (Model_task): Specifies whether it"s a "CLASSIFICATION_TASK" or "REGRESSION_TASK".
            n_estimators (int, optional): The number of weak learners in the ensemble (default: 1000).
            learning_rate (float, optional): Controls the contribution of each weak learner (default: 0.1).
        """
        super().__init__(model_task)
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate

    def construct(self):
        """
        Constructs the AdaBoost model. Selects "AdaBoostClassifier" for classification
        tasks and "AdaBoostRegressor" for regression tasks.
        """
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = AdaBoostClassifier(learning_rate=self._learning_rate,
                                            n_estimators=self._n_estimators)
        else:
            self.model = AdaBoostRegressor(learning_rate=self._learning_rate,
                                           n_estimators=self._n_estimators)

    def save(self, file_path_and_name):
        """
        Saves the trained AdaBoost model to a file (automatically adds ".adab" extension).

        Args:
            file_path_and_name (str): The path and filename where the model should be saved.
                                      (automatically adds ".adab" extension).
        """
        super().save(file_path_and_name, ".adab")

    def load(self, file_path_and_name):
        """
        Loads a previously saved AdaBoost model from a file.

        Args:
            file_path_and_name (str): The path and filename of the saved model.
                                      (automatically adds ".adab" extension).
        """
        super().load(file_path_and_name, ".adab")


class RandomForest_model(Sklearn_model):
    """
    A class representing a Random Forest ensemble model using scikit-learn.
    Random Forests construct multiple decision trees for improved prediction.
    """

    def __init__(self, model_task, n_estimators=1000, max_depth=None, n_jobs=50):
        """
        Initializes a RandomForest_model instance.

        Args:
            model_task (Model_task): Specifies whether it"s a "CLASSIFICATION_TASK" or "REGRESSION_TASK".
            n_estimators (int, optional): The number of decision trees in the forest (default: 1000).
            max_depth (int, optional): The maximum depth of each decision tree (default: None).
            n_jobs (int, optional): The number of jobs to run in parallel (default: 50).
        """
        super().__init__(model_task)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._n_jobs = n_jobs

    def construct(self):
        """
        Constructs the Random Forest model. Selects "RandomForestClassifier" for classification
        tasks and "RandomForestRegressor" for regression tasks.
        """
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = RandomForestClassifier(max_depth=self._max_depth,
                                                n_estimators=self._n_estimators,
                                                n_jobs=self._n_jobs)
        else:
            self.model = RandomForestRegressor(max_depth=self._max_depth,
                                               n_estimators=self._n_estimators,
                                               n_jobs=self._n_jobs)

    def save(self, file_path_and_name):
        """
        Saves the trained Random Forest model to a file.

        Args:
            file_path_and_name (str): The path and filename where the model should be saved.
                                      (automatically adds ".rf" extension).
        """
        super().save(file_path_and_name, ".rf")

    def load(self, file_path_and_name):
        """
        Loads a previously saved Random Forest model from a file.

        Args:
            file_path_and_name (str): The path and filename of the saved model.
                                      (automatically adds ".rf" extension).
        """
        super().load(file_path_and_name, ".rf")


class Lasso_model(Sklearn_model):
    """
    A class representing a Lasso regression model using scikit-learn.
    Note: For classification, this uses LogisticRegression with L1 penalty.
    """

    def __init__(self, model_task, max_iter=1000, alpha=1, reg_penalty="l1", reg_solver="saga"):
        """
        Initializes a Lasso model instance.

        Args:
            model_task (Model_task): Specifies whether it is a "CLASSIFICATION_TASK" or "REGRESSION_TASK".
            max_iter (int, optional): Maximum number of iterations for the solver (default: 1000).
            alpha (float, optional): Regularization strength (default: 1).
            reg_penalty (str, optional): Type of regularization see scikit-learn API for other options (default: "l1").
            reg_solver (str, optional): Solver to use, see scikit-learn API for other options (default: "saga").
        """
        super().__init__(model_task)
        self._max_iter = max_iter
        self._reg_penalty = reg_penalty
        self._reg_solver = reg_solver
        self._alpha = alpha

    def construct(self):
        """
        Constructs the Lasso model.  For classification, this uses LogisticRegression
        with L1 penalty to approximate Lasso-like behavior.
        """
        if self._model_task == Model_task.CLASSIFICATION_TASK:
            self.model = linear_model.Lasso(max_iter=self._max_iter, alpha=self._alpha)
        else:
            # Note: I using here the Logistic Regression model with L1 penalty
            # "lbfgs"- the default - ["l2", "none"], "saga" - ["elasticnet", "l1", "l2", "none"]
            # for small datasets, ‘liblinear’ is a good choice,
            # whereas ‘sag’ and ‘saga’ are faster for large ones;
            self.model = linear_model.LogisticRegression(
                max_iter=self._max_iter, penalty=self._reg_penalty, solver=self._reg_solver)  # type: ignore

    def save(self, file_path_and_name):
        """
        Saves the trained Lasso model to a file.

        Args:
            file_path_and_name (str): The path and filename where the model should be saved.
                                      (automatically adds ".lasso" extension).
        """
        super().save(file_path_and_name, ".lasso")

    def load(self, file_path_and_name):
        """
        Loads a previously saved Lasso model from a file.

        Args:
            file_path_and_name (str): The path and filename of the saved model.
                                      (automatically adds ".lasso" extension).
        """
        super().load(file_path_and_name, ".lasso")


class MLP_model(NN_model):
    def __init__(self, model_task, batch_size=32, epochs=10, learning_rate=0.1,
                 dense_layers=(128,), activation_funs=("relu",), optimizer="adam",
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
