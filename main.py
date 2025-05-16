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
        self.h = np.zeros()

    def calculate_loss(self, y_hat):
        loss = 0
        for i in range(len(y_hat)):
            loss += -np.log(y_hat[i])

    def backpropagate(self, inputs, targets, y_hat):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_yh = np.zeros_like(self.W_yh)
        db_h = np.zeros_like(self.b_h)
        db_o = np.zeros_like(self.b_o)

        for i in range(len()):
            db_o +=

if __name__ == '__main__':
    tau = 10
    input_size = tau
    output_size = tau
    hidden_size = tau - 1


