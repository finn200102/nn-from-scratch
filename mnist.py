import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensor import Tensor
import time

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def mnist_one_layer_net(epochs=5, learning_rate=0.01):
    # Load and preprocess data
    mnist = fetch_openml('mnist_784')
    X, y = mnist.data, mnist.target.astype(int)
    X = StandardScaler().fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_test = y_test.to_numpy()
    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train, 10)

    # Initialize weights
    input_size = X_train.shape[1]
    output_size = 10
    hidden_size = 100
    W = Tensor(np.random.randn(hidden_size, input_size) * 0.01)
    W2 = Tensor(np.random.randn(output_size, hidden_size) * 0.01)
    b = Tensor(np.zeros(hidden_size))
    b2 = Tensor(np.zeros(output_size))

    # Training loop
    for epoch in range(epochs):
        t1 = time.time()
        total_loss = 0
        for i in range(len(X_train)):
            # Forward pass
            x = Tensor(X_train[i])  # Shape: (784,)
            y_true = Tensor(y_train_onehot[i])  # Shape: (10,)

            z1 = W @ x + b  # Shape: (10,)
            a1 = z1.relu()
            z = W2 @ a1 + b2
            y_pred = z.softmax()
            # Compute loss
            y_pred_log = y_pred.log()
            y_prod = y_true * y_pred_log
            y_prod_sum = y_prod.sum()
            loss = Tensor(-1) * y_prod_sum

            total_loss += loss.value

            # Backward pass
            loss.backprop()

            # Update weights
            W.value -= learning_rate * W.grads
            b.value -= learning_rate * b.grads

            # Clear gradients
            loss.clear()
        t2 = time.time()
        T = t2-t1
        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(X_train):.20f}---time:{T}")

    # Evaluate on test set
    correct = 0
    for i in range(len(X_test)):

        x = Tensor(X_test[i])  # Shape: (784,)
        z1 = W @ x + b  # Shape: (10,)
        a1 = z1.relu()
        z = W2 @ a1 + b2
        y_pred = z.softmax()
        if np.argmax(y_pred.value) == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    print(f"Test Accuracy: {accuracy:.4f}")

# Run the network
mnist_one_layer_net()

