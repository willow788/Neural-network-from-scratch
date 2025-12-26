#from scatch we're making the neural network
#why beause we're cool like that


#author: Srija (willow788)
#co author: chatgpt (full stack developer)

import numpy as np

np.random.seed(42)
#fix randomness for repeatability

#defining the activation function
#now we're using sigmoid function
#but we can use relu or softmax too

#lets do sigmoid 1st

def sigmoid_functn(x):
    return 1/(1+ np.exp(-x))

def sigmoid_Derivative(x):
    #x is here sigmoid output only
    return x * (1-x) 
#negative learning rate

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)


input_neurons = 2
hidden_neurons = 4
output_neurons = 1
lr = np.float32(0.5)

#now we will innitialize weights and biases

w1 = (np.random.randn(input_neurons, hidden_neurons) * 0.1).astype(np.float32)
b1 = np.zeros((1, hidden_neurons), dtype=np.float32)

w2 = (np.random.randn(hidden_neurons, output_neurons) * 0.1).astype(np.float32)
b2 = np.zeros((1, output_neurons), dtype=np.float32)

#weight is how much importance we give to a particular input
#bias is used to shift the activation function


#training loop
max_epochs = 20000
early_stop_loss = 1e-3
for epoch in range(max_epochs):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid_functn(z1)

    z2 = np.dot(a1, w2) + b2
    y_hat = sigmoid_functn(z2)

    #loss calculation
    loss = np.mean((y - y_hat) ** 2)

    #backward pass (everything stays vectorized)
    d_loss_yhat = (y_hat - y)
    d_yhat_z2 = sigmoid_Derivative(y_hat)
    d_z2 = d_loss_yhat * d_yhat_z2

    dw2 = np.dot(a1.T, d_z2)
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, w2.T)
    d_a1_z1 = sigmoid_Derivative(a1)
    d_z1 = d_a1 * d_a1_z1
    dw1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    #updating weights and biases in-place keeps things fast
    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    if loss <= early_stop_loss:
        print(f'Early stop at epoch {epoch}, Loss: {loss:.4f}')
        break

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

print("Final output after training:")
print(y_hat)
