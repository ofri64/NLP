class POSTaggerInterface:

    def fit(self, x_train, y_train, callbacks=None):
        """
        Train the model over the given labeled data set
        :param x_train: np.ndarray of shape (num_samples, max_input_length, vocab_size)
        :param y_train: np.ndarray of shape (num_samples, max_input_length, n_classes)
        :param callbacks: Keras callback object to use within training
        """
        raise NotImplementedError("Fit method is not implemented")

    def evaluate_sample(self, x_test, y_test):
        """
        Evaluate the model's accuracy over a given test set
        :param x_test: np.ndarray of shape (num_samples, max_input_length, vocab_size)
        :param y_test: np.ndarray of shape (num_samples, max_input_length, n_classes)
        :return: accuracy score
        """
        raise NotImplementedError("Evaluate sample method is not implemented")

    def evaluate_sample_conditioned(self, x_test, y_test, condition):
        """
        Evaluate the model's accuracy given a specific condition over the given test set
        :param x_test: np.ndarray of shape (num_samples, max_input_length, vocab_size)
        :param y_test: np.ndarray of shape (num_samples, max_input_length, n_classes)
        :param condition: String one of {unseen, ambiguous}
        :return: accuracy score
        """
        raise NotImplementedError("Evaluate sample conditioned method is not implemented")

    def predict_sentence(self, sentence):
        """
        Predict POS tags for a given sentence
        :param sentence: np.ndarray of shape (1, max_input_length, vocab_size)
        :return: np.ndarray of shape (1, sentence_length)
        """
        raise NotImplementedError("Predict sentence conditioned method is not implemented")

    def save_model_params(self):
        """
        Save the model attributes to a file.
        """
        raise NotImplementedError("Save model parameters method is not implemented")

    def load_model_params(self, file_path):
        """
        Load the model attributes from a file.
        """
        raise NotImplementedError("Load model parameters method is not implemented")