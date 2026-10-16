import numpy as np

class NeuralNetwork:
    """
    Fully connected neural network for binary classification.

    Architecture:
        Input -> Hidden Layers -> Output layer
        Input -> Dense + ReLU -> Dense + Sigmoid

    Input shape:  (n_samples, n_features)
    Target shape: (n_samples, 1)
    """

    def __init__(
        self,
        input_size,
        hidden_size=8,
        output_size=1,
        learning_rate=0.1,
        seed=3,
    ):

        # Validate inputs
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("Layer sizes must be positive.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)  # Pass an integer seed for reproducibility

        # Initialization for hidden layer
        self.W1 = rng.standard_normal((input_size, hidden_size))
        # He initialization for ReLU hidden layer.
        # self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # Hidden -> output layer
        self.W2 = rng.standard_normal((hidden_size, output_size))
        # Xavier initialization for sigmoid output layer.
        # self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def sigmoid(z):
        # Clipping prevents numerical overflow in exp().
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):

        # Clip predictions to prevent log(0) or log(1)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        return -np.mean(
            y_true * np.log(y_pred)
            + (1.0 - y_true) * np.log(1.0 - y_pred)
        )

    def forward(self, X):
        """Feedforward: input -> hidden -> output."""

        self.X = X

        # Hidden layer.
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        # Output layer.
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def backward(self, y_true):
        """Backpropagation: compute gradients using the chain rule."""

        m = y_true.shape[0]

        # Sigmoid + binary cross-entropy derivative.
        dZ2 = (self.A2 - y_true) / m

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Propagate error to hidden layer.
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)

        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

    def update_parameters(self, gradients):
        """Gradient descent parameter update."""

        lr = self.learning_rate

        self.W1 -= lr * gradients["dW1"]
        self.b1 -= lr * gradients["db1"]

        self.W2 -= lr * gradients["dW2"]
        self.b2 -= lr * gradients["db2"]

    def fit(self, X, y, epochs=5000, log_every=500):
        """Train using full-batch gradient descent."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError("X must have shape (n_samples, input_size).")
        if X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("X and y must have the same nonzero sample count.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain finite values.")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Binary targets must be 0 or 1.")
        if epochs <= 0 or log_every < 0:
            raise ValueError("Invalid epochs or log_every.")

        history = []

        for epoch in range(1, epochs + 1):
            # 1. Forward pass.
            probabilities = self.forward(X)

            # 2. Calculate loss.
            loss = self.binary_cross_entropy(y, probabilities)
            history.append(loss)

            # 3. Backward pass.
            gradients = self.backward(y)

            # 4. Update weights and biases.
            self.update_parameters(gradients)

            if log_every and (
                epoch == 1
                or epoch % log_every == 0
                or epoch == epochs
            ):
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        return history

    def predict_proba(self, X):
        """Return class-1 probabilities."""

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError("X has the wrong shape.")

        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """Return binary class predictions."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        return (self.predict_proba(X) >= threshold).astype(int)

if __name__ == "__main__":
    # Simple binary classification dataset.
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.3],
        [0.3, 0.8],
        [0.8, 0.2],
        [0.9, 0.9],
    ])

    # Class 1 when x1 + x2 > 1.
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])

    model = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        learning_rate=0.1,
        seed=42,
    )

    model.fit(X, y, epochs=5000, log_every=1000)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    print("\nProbabilities:")
    print(np.round(probabilities.ravel(), 4))

    print("\nPredictions:")
    print(predictions.ravel())

    print("\nActual labels:")
    print(y)

    accuracy = np.mean(predictions.ravel() == y)
    print(f"\nTraining accuracy: {accuracy:.2%}")