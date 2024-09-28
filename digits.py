import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensor import Tensor  # Assuming your Tensor class is in a module

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def mnist_one_layer_net(epochs=10, learning_rate=0.001):
    # Load and preprocess data
    digits = load_digits()
    X, y = digits.data, digits.target
    print(X.shape)
    X = StandardScaler().fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train, 10)

    # Initialize weights
    input_size = X_train.shape[1]
    output_size = 10
    W = Tensor(np.random.randn(output_size, input_size) * 0.01)
    b = Tensor(np.zeros(output_size))

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_train)):
            # Forward pass
            x = Tensor(X_train[i])  # Shape: (64,)
            y_true = Tensor(y_train_onehot[i])  # Shape: (10,)
 
            z = W @ x + b  # Shape: (10,)
            y_pred = z.softmax()
            # Compute loss
            y_pred_log = y_pred.log()
            y_prod = y_true * y_pred_log
           # print(y_prod.value)
            y_prod_sum = y_prod.sum()
            loss = Tensor(-1) * y_prod_sum

            total_loss += loss.value

            # Backward pass
            loss.backprop()

           # print(loss.grads)
           # print(y_prod_sum.grads)
           # print(y_prod.grads)
            
           # print(y_pred_log.grads)

            # Update weights
            W.value -= learning_rate * W.grads
            b.value -= learning_rate * b.grads

            # Clear gradients
            loss.clear()
        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(X_train):.20f}")

    # Evaluate on test set
    correct = 0
    for i in range(len(X_test)):
        x = Tensor(X_test[i])  # Shape: (64,)
        z = W @ x + b  # Shape: (10,)
        y_pred = z.exp() / z.exp().sum()
        if np.argmax(y_pred.value) == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    print(f"Test Accuracy: {accuracy:.4f}")

# Run the network
mnist_one_layer_net()

