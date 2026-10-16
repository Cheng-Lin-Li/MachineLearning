# Feedforward Neural Network + Backpropagation in Python for binary classification

For an interview, the strongest implementation is a fully connected neural network from scratch using NumPy. 
This example code demonstrates the understanding of the forward pass, loss calculation, backpropagation, gradient descent, and the chain rule.

This example code implements a network with:

- Feedforward: Computes predictions layer by layer.
- Loss: Binary cross-entropy for binary classification.
- Backpropagation: Computes gradients using the chain rule.
- Gradient descent: Updates weights and biases.
- Training: Repeatedly performs forward and backward passes.
- Prediction: Converts probabilities into class labels.

## 1. Understand the architecture

Input layer

$X$: 2 features

Hidden layer

4 neurons · ReLU

Output layer

1 neuron · Sigmoid

Prediction

Probability of class 1

The forward pass moves downward through the network. Backpropagation computes gradients in the reverse direction.

The mathematical operations for each layer are:

$$
Z^{[l]}=A^{[l-1]}W^{[l]}+b^{[l]}
$$

$$
A^{[l]}=g^{[l]}(Z^{[l]})
$$

Where $W$ represents weights, $b$ represents biases, $Z$ represents pre-activation values, and $A$ represents activations.

## 2. Complete implementation from scratch

Install NumPy first:

bash
```
pip install numpy
```

Save the following as `neural_network.py`.

python
```

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
        seed=42,
    ):
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("Layer sizes must be positive.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)

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
```

## 3. How backpropagation works

This is the most important part to explain in an interview.

### Step 1: Feedforward

For input $X$, hidden-layer weights $W_1$, and output weights $W_2$:

$$
Z_1=XW_1+b_1
$$

$$
A_1=\operatorname{ReLU}(Z_1)
$$

$$
Z_2=A_1W_2+b_2
$$

$$
\hat{Y}=\sigma(Z_2)
$$

The network produces a probability between 0 and 1.

### Step 2: Calculate the loss

Binary cross-entropy measures how far the predicted probabilities are from the true labels.

$$
L=-\frac{1}{m}\sum_{i=1}^{m}
[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)]
$$

A lower loss generally indicates better predictions on the training data.

### Step 3: Compute output-layer gradients

For sigmoid combined with binary cross-entropy, the derivative simplifies to:

$$
dZ_2=\frac{\hat Y-Y}{m}
$$

Then:

$$
dW_2=A_1^T dZ_2
$$

$$
db_2=\sum_{i=1}^{m}dZ_2^{(i)}
$$

This simplification is a particularly useful interview detail.

### Step 4: Propagate the error backward

First, propagate the output error to the hidden layer:

$$
dA_1=dZ_2W_2^T
$$

The ReLU derivative is 1 for positive inputs and 0 otherwise:

$$
dZ_1=dA_1\odot\mathbf{1}(Z_1>0)
$$

Then compute the hidden-layer gradients:

$$
dW_1=X^TdZ_1
$$

$$
db_1=\sum_{i=1}^{m}dZ_1^{(i)}
$$

The symbol $\odot$ denotes element-wise multiplication.

### Step 5: Update parameters

Gradient descent moves the weights in the direction that reduces the loss:

$$
W\leftarrow W-\eta dW
$$

$$
b\leftarrow b-\eta db
$$

Here, $\eta$ is the learning rate.

## 4. Interview questions to prepare

| Question | Key answer |
| --- | --- |
| What is feedforward propagation? | Computing activations and predictions from input to output. |
| What is backpropagation? | Applying the chain rule to compute loss gradients for each parameter. |
| Why use ReLU? | It is computationally simple and helps mitigate vanishing gradients in hidden layers. |
| Why use sigmoid at the output? | It produces a probability for binary classification. |
| Why divide gradients by $m$? | To calculate the average gradient across the batch. |
| Why initialize weights randomly? | To break symmetry between neurons. |
| Why use He initialization? | It helps preserve activation variance in ReLU networks. |
| What happens if the learning rate is too high? | Training can become unstable or diverge. |
| What is the difference between an epoch and an iteration? | An epoch processes the entire training set; an iteration performs one parameter update. |

## 5. What makes this implementation interview-ready?

To take it a step further, be prepared to implement or discuss:

- Numerical gradient checking: Compare backpropagation gradients with finite-difference approximations.
- Mini-batch gradient descent: Update parameters using small subsets of the training data.
- Multiclass classification: Replace the sigmoid output with softmax and use categorical cross-entropy.
- Regularization: Add L2 penalties to reduce overfitting.
- Vectorization: Use NumPy matrix operations instead of loops over individual samples and neurons.

One important distinction: This implementation uses full-batch gradient descent and supports one hidden layer. The same principles generalize to deeper networks, where backpropagation applies the chain rule recursively through each layer.