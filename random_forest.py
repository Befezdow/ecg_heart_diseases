from sklearn import ensemble
import numpy as np


class RFClassifier:
    def __init__(self, estimators_number, batch_size=1000):
        self.model = ensemble.RandomForestClassifier(n_estimators=estimators_number, oob_score=True)

        self.batch_size = batch_size

    def fit(self, x_train, y_train):
        batch_size = self.batch_size
        dataset_size = x_train.shape[0]
        batches_count = int(np.ceil(dataset_size / batch_size))

        for batch_index in range(batches_count):
            x_var = x_train.iloc[batch_index * batch_size:batch_index * batch_size + batch_size, :]
            y_var = y_train.iloc[batch_index * batch_size:batch_index * batch_size + batch_size]
            self.model.fit(x_var, y_var)

        return np.mean(self.model.predict(x_train) != y_train)

    def check(self, x, y):
        batch_size = self.batch_size
        dataset_size = x.shape[0]
        batches_count = int(np.ceil(dataset_size / batch_size))

        loss = 0
        for batch_index in range(batches_count):
            x_var = x.iloc[batch_index * batch_size:batch_index * batch_size + batch_size, :]
            y_var = y.iloc[batch_index * batch_size:batch_index * batch_size + batch_size]
            loss += np.mean(self.model.predict(x_var) != y_var)

        loss /= batches_count
        return loss

    def predict(self, x):
        return self.model.predict(x)

    @property
    def oob_error(self):
        return 1 - self.model.oob_score_
