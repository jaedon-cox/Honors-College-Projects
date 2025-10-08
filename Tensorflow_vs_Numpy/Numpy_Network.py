import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Layer_Dense:
    def __init__(self, input_size, nodes, activ):
        self.weights = np.random.rand(input_size, nodes) * 2 - 1
        self.biases = np.zeros((1, nodes))
        if activ == "relu":
            self.activation = Activation_ReLU()
        elif activ == "softmax":
            self.activation = Activation_Softmax()
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(self.output)
    def backward(self, dvalues):
        if isinstance(self.activation, Activation_ReLU):
            dvalues = self.activation.backward(dvalues)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
    
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
class Activation_Softmax_Loss_Cross_Entropy:
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Cross_Entropy_Loss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
class Neural_Network:
    def __init__(self, input_size):
        self.layers = []
        self.previous_size = input_size
    def add_layer(self, size, activ):
        self.layers.append(Layer_Dense(self.previous_size, size, activ))  
        self.previous_size = size
    def process(self, input):
        self.output = input
        for layer in self.layers:
            layer.forward(self.output)
            self.output = layer.output
        return self.output
    def calculate_loss(self, expected):
        self.loss_function = Cross_Entropy_Loss()
        return self.loss_function.calculate(self.output, expected)
    def backward(self, expected):
        loss_activation = Activation_Softmax_Loss_Cross_Entropy()
        dvalues = loss_activation.backward(self.output, expected)

        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs
    def train(self, x_train, y_train, num_trains, train_weight):
        batch_size = 32
        for epoch in range(num_trains):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                self.process(x_batch)
                loss = self.calculate_loss(y_batch)
                self.backward(y_batch)

                for layer in self.layers:
                    layer.weights -= train_weight * layer.dweights
                    layer.biases -= train_weight * layer.dbiases
            
            if epoch % 1 == 0:
                print(f"Epoch: {epoch+1}, Loss: {loss}")
    def predict(self, x_input):
        self.process(x_input)
        return np.argmax(self.output, axis=1)



nn = Neural_Network(28*28)
nn.add_layer(128, "relu")
nn.add_layer(128, "relu")
nn.add_layer(10, "softmax")

mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], -1)

nn.train(x_train, y_train, 10, 0.02)

image_num = 1
while os.path.isfile(f'digits/digit{image_num}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_num}.png')[:,:,0]
        img = np.invert(np.array([img]))
        img_vector = img.reshape(1, -1)
        prediction = nn.predict(img_vector)
        print(f'This digit is likely a {prediction[0]}')
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print('Error!')
    finally:
        image_num += 1

