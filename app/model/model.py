class my_model:
    def __init__(self):
        self.model = None
        self.data = None

    def data(self):
        return self.data

    def train(self, data):
        self.data = data
        self.model = train_model(data)

    def predict(self, data):
        return predict_model(self.model, data)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
