import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Perceptron:
    """
    Perceptron classifier.
    Parameters
    ----------
    eta : float
        Learning rate (0.0 < eta <= 1.0)
    n_iter : int
        Passes over the training dataset (epochs).
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data."""
        self.w_ = np.zeros(1 + X.shape[1])  # weights
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class AdalineGD:
    """
    ADAptive LInear NEuron classifier.
    Parameters
    ----------
    eta : float
        Learning rate (0.0 < eta <= 1.0)
    n_iter : int
        Passes over the training dataset (epochs).
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data."""
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# ------------------- DEMO --------------------
if __name__ == "__main__":
    # Load dataset (Iris)
    iris = load_iris()
    X = iris.data[:100, [0, 2]]  # use first 100 samples (setosa vs versicolor)
    y = iris.target[:100]
    y = np.where(y == 0, -1, 1)  # convert to -1 and 1

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train Perceptron
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X_train, y_train)

    print("Perceptron Weights:", ppn.w_)
    print("Perceptron Training Errors:", ppn.errors_)

    # Train Adaline
    ada = AdalineGD(eta=0.01, n_iter=20)
    ada.fit(X_train, y_train)

    print("Adaline Weights:", ada.w_)
    print("Adaline Costs:", ada.cost_)

    # Plot errors/costs
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.title('Perceptron - Training Errors')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.title('Adaline - Training Cost')

    plt.tight_layout()
    plt.show()
