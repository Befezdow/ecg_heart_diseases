from sklearn import ensemble
import numpy as np


class RFClassifier:
    def __init__(self, estimators_number, batch_size=1000):
        self.model = ensemble.RandomForestClassifier(n_estimators=estimators_number, oob_score=True)

        self.batch_size = batch_size

    def fit(self, x_train, y_train, lr_logger=None, loss_logger=None):
        batch_size = self.batch_size
        dataset_size = x_train.shape[0]
        batches_count = int(np.ceil(dataset_size / batch_size))

        for batch_index in range(batches_count):
            x_var = x_train.iloc[batch_index * batch_size:batch_index * batch_size + batch_size, :]
            y_var = y_train.iloc[batch_index * batch_size:batch_index * batch_size + batch_size]
            self.model.fit(x_var, y_var)

        return np.mean(self.model.predict(x_train) != y_train), 0

    def check(self, x, y):
        batch_size = self.batch_size
        dataset_size = x.shape[0]
        batches_count = int(np.ceil(dataset_size / batch_size))

        error = 0
        for batch_index in range(batches_count):
            x_var = x.iloc[batch_index * batch_size:batch_index * batch_size + batch_size, :]
            y_var = y.iloc[batch_index * batch_size:batch_index * batch_size + batch_size]
            error += np.mean(self.model.predict(x_var) != y_var)

        error /= batches_count
        return error, 0

    def predict(self, x):
        return self.model.predict(x)

    @property
    def oob_error(self):
        return 1 - self.model.oob_score_
