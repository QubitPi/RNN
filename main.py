import numpy as np


def random_matrix(num_rows, num_cols):
    """
    Initializes a random matrix of size n x m.

    Args:
      n: Number of rows.
      m: Number of columns.

    Returns:
      A NumPy array representing the random matrix.
    """
    return np.random.rand(num_rows, num_cols)

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_xh = random_matrix(num_rows=hidden_size, num_cols=input_size) * 0.01
        self.W_hh = random_matrix(num_rows=hidden_size, num_cols=hidden_size) * 0.01
        self.W_yh = random_matrix(num_rows=output_size, num_cols=hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((output_size, 1))

    def batch_inference(self, inputs):
        # Since we are inferencing in batch, h will be an array of matrices, i.e. a 3-dimensional object
        # Each matrix has a dimension of (n x 1)
        # And since h has an initial hidden state h_0, its length will be a "+1"
        self.h = np.zeros((len(inputs) + 1, self.W_hh.shape[0], 1))

        self.o = np.zeros((len(inputs), self.W_yh.shape[0], 1))
        self.y_hat = np.zeros((len(inputs), self.W_yh.shape[0], 1))

        for idx, input in enumerate(inputs):
            input = input.reshape(-1, 1) # transpose row-vector to column-vector

            # since idx starts with 0, we should have (idx v.s. idx + 1) instead of the original (idx v.s. idx - 1)
            self.h[idx + 1] = np.tanh(np.dot(self.W_hh, self.h[idx]) + np.dot(self.W_xh, input) + self.b_h)
            self.o[idx] = np.dot(self.W_yh, self.h[idx + 1]) + self.b_o
            self.y_hat[idx] = np.exp(self.o[idx]) / np.sum(np.exp(self.o[t]))

        return self.y_hat

    def calculate_loss(self, y_hat):
        loss = 0
        for i in range(len(y_hat)):
            loss += -np.log(y_hat[i])

    def backpropagate(self, inputs, targets):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_yh = np.zeros_like(self.W_yh)
        db_h = np.zeros_like(self.b_h)
        db_o = np.zeros_like(self.b_o)

        for i in range(len(inputs)):
            dy = np.copy(self.y_hat[i])
            db_o += dy
            dW_yh += np.dot(dy, self.h[i + 1].T)


if __name__ == '__main__':
    tau = 10
    input_size = tau
    output_size = tau
    hidden_size = tau - 1
