import os
import time


class TensorflowAbstractModel(object):
    """
    An Abstract tensorflow graph class
    """

    def __init__(self):
        """
        Define placeholders and variable tensors as object members
        Subclass object is responsible for calling the build_graph() method
        after initiating other model variables if needed
        """
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None
        self.prediction_variable_tensor = None
        self.loss_tensor = None
        self.training_op_tensor = None

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input tensors
        """
        raise NotImplementedError("Each model must implement this method")

    def create_feed_dict(self, input_batch, mask_batch, labels_batch=None, dropout=0.5):
        raise NotImplementedError("Each model must implement this method")

    def add_prediction_op(self):
        """
        Implements the core of the model - the transformation between a batch of input data
        into a matching size batch of predictions vectors
        :return:
        pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each model must implement this method")

    def add_loss_op(self, predictions):
        """
        Adds op for computing the loss function
        :param predictions: a tensor of shape (batch_size, n_classes)
        :return: a 0-d tensor (scalar) representing the loss
        """
        raise NotImplementedError("Each model must implement this method")

    def add_training_op(self, loss):
        """
        Sets up an optimizer and applies the gradients to all trainable variables
        the op returned by this function will be passed to 'session.run()' call in order to
        train the model
        :param loss: 0-d tensor (scalar) representing the loss
        :return: tf op (specifically optimizer) object
        """
        raise NotImplementedError("Each model must implement this method")

    def build_graph(self):
        """
        Build the components of the computation graph before applying a session
        """
        self.add_placeholders()
        self.prediction_variable_tensor = self.add_prediction_op()
        self.loss_tensor = self.add_loss_op(self.prediction_variable_tensor)
        self.training_op_tensor = self.add_training_op(self.loss_tensor)

    def train_on_batch(self, session, input_batch, labels_batch, mask_batch):
        """
        Perform one step of gradient descent on the provided batch of data
        :param session: tf.Session object
        :param input_batch: np.ndarray object of shape (n_samples, n_features)
        :param labels_batch: np.ndarray object of shape (n_samples, n_classes)
        :param mask_batch: Boolean array in case of a batch recurrent model
        :return: average loss over the batch (a scalar)
        """
        feed_dict = self.create_feed_dict(input_batch, labels_batch=labels_batch, mask_batch=mask_batch)
        _, loss = session.run([self.training_op_tensor, self.loss_tensor], feed_dict=feed_dict)
        return loss

    def predict_on_batch(self, session, input_batch, mask_batch):
        """
        Make a prediction over the provided batch.
        Don't apply gradient update steps just feed forward over the graph
        :param session: tf.Session object
        :param input_batch: np.ndarray of shape (n_samples, n_features)
        :param mask_batch: Boolean array in case of a batch recurrent model
        :return: np.ndarray of shape (n_samples, n_classes)
        """
        feed_dict = self.create_feed_dict(input_batch, mask_batch)
        predictions = session.run(self.prediction_variable_tensor, feed_dict=feed_dict)
        return predictions

    @staticmethod
    def get_save_path(model_prefix):
        current_time_date = time.strftime("%x") + "-" + time.strftime("%X")
        current_folder = os.curdir
        save_folder = os.path.join(current_folder, current_time_date)
        model_prefix = model_prefix

        # create directory
        os.mkdir(save_folder)
        save_path = os.path.join(save_folder, model_prefix)
        return save_path
