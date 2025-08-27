import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, layer_dims, activation_func=None, activation_deriv=None):
        self.L = len(layer_dims) - 1
        self.layer_dims = layer_dims
        self.parameters = {}
        self.g = activation_func if activation_func is not None else self.sigmoid
        self.g_prime = activation_deriv if activation_deriv is not None else self.sigmoid_derivative

        for l in range(1, self.L + 1):
            # Using He Initialization
            self.parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def feed_forward(self, A0):
        cache = {"A0": A0}
        A = A0
        for l in range(1, self.L + 1):
            W, b = self.parameters[f'W{l}'], self.parameters[f'b{l}']
            Z = W @ A + b
            A = self.g(Z)
            cache[f'Z{l}'], cache[f'A{l}'] = Z, A
        return A, cache

    def compute_cost(self, y_hat, Y):
        m = Y.shape[1]
        cost = - (1/m) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
        return np.squeeze(cost)

    def backpropagate(self, y_hat, Y, cache):
        grads = {}
        m = Y.shape[1]
        dC_dZL = y_hat - Y
        for l in reversed(range(1, self.L + 1)):
            dC_dZ = dC_dZL if l == self.L else dC_dZ
            A_prev = cache[f'A{l-1}']
            grads[f'dC_dW{l}'] = (1/m) * (dC_dZ @ A_prev.T)
            grads[f'dC_db{l}'] = (1/m) * np.sum(dC_dZ, axis=1, keepdims=True)
            if l > 1:
                W = self.parameters[f'W{l}']
                A = cache[f'A{l-1}']
                dC_dA_prev = W.T @ dC_dZ
                dZ_dA_prev = self.g_prime(A)
                dC_dZ = dC_dA_prev * dZ_dA_prev
        return grads

    def update_parameters(self, grads, alpha):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= alpha * grads[f'dC_dW{l}']
            self.parameters[f'b{l}'] -= alpha * grads[f'dC_db{l}']

    def train(self, X, Y, epochs=1000, alpha=0.1, print_cost_every=100):
        costs = []
        for e in range(epochs):
            y_hat, cache = self.feed_forward(X)
            cost = self.compute_cost(y_hat, Y)
            if e % print_cost_every == 0 or e == epochs - 1:
                print(f"Epoch {e}: cost = {cost:.6f}")
                costs.append(cost)
            grads = self.backpropagate(y_hat, Y, cache)
            self.update_parameters(grads, alpha)
        print("Training complete.")
        return costs
    
    def predict(self, X_new):
        """
        Makes a prediction on new data.

        Args:
            X_new (np.array): New input data, shape (n_features, n_examples).

        Returns:
            np.array: A binary prediction (0 or 1).
        """
        # Run the feed forward pass to get the probability
        probability, _ = self.feed_forward(X_new)
        
        # Convert probability to a 0 or 1 prediction using a 0.5 threshold
        predictions = (probability > 0.5).astype(int)
        
        return predictions

if __name__ == '__main__':
    # Prepare data
    df = pd.read_csv('train.csv')

    # Prepare data
    X_train = df.drop('HeartDisease', axis=1)
    y_train = df['HeartDisease']
    
    # SCALE THE DATA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    A0 = X_scaled.T
    Y = y_train.to_numpy().reshape(1, y_train.shape[0])

    layer_dimensions = [A0.shape[0], 5, 4, 1] 
    
    nn = NeuralNetwork(layer_dims=layer_dimensions)
    
    costs = nn.train(A0, Y, epochs=10000, alpha=0.1, print_cost_every=100)
    
    # Plot cost
    plt.plot(costs)
    plt.title("Cost Function After Improvements")
    plt.xlabel(f"Epochs")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

    # Example prediction
    new_patient = np.array([[300, 50]])
    new_patient_scaled = scaler.transform(new_patient)
    prediction = nn.predict(new_patient_scaled.T)
    print("\n--- New Patient Prediction ---")
    if prediction[0][0] == 1:
        print("Prediction: The model predicts this patient WILL have heart disease. ❤️")
    else:
        print("Prediction: The model predicts this patient WILL NOT have heart disease. ✅")

