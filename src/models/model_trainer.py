from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model
