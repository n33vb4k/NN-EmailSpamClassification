import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


train_data = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
test_data = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)

all_data = np.concatenate((test_data, train_data), axis=0)
#np.random.shuffle(all_data)
#split into 86.6%training : 13.4%testing
train = all_data[:1300].T
test = all_data[1300:].T

X_train = train[1:]
Y_train = train[0]
X_test = test[1:]
Y_test = test[0]


class SpamClassifierNN:
    def __init__(self, input_size=54, hidden_size1=54, hidden_size2=54, output_size=2): #20, 10
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.W1 = np.random.randn(self.hidden_size1, self.input_size)   * 0.01  # weight matrix from input to hidden layer 1
        self.W2 = np.random.randn(self.hidden_size2, self.hidden_size1) * 0.01  # weight matrix from hidden layer 1 to hidden layer 2
        self.W3 = np.random.randn(self.output_size, self.hidden_size2)  * 0.01  # weight matrix from hidden layer 2 to output
        
        self.b1 = np.zeros((self.hidden_size1, 1))      # bias for hidden layer 1
        self.b2 = np.zeros((self.hidden_size2, 1))      # bias for hidden layer 2
        self.b3 = np.zeros((self.output_size, 1))       # bias for output layer


    def save_model(self, file_name):
        np.savez(file_name, W1=self.W1, W2=self.W2, W3=self.W3, b1=self.b1, b2=self.b2, b3=self.b3)


    def load_model(self, file_name):
        npzfile = np.load(file_name)
        self.W1 = npzfile['W1']
        self.W2 = npzfile['W2']
        self.W3 = npzfile['W3']
        self.b1 = npzfile['b1']
        self.b2 = npzfile['b2']
        self.b3 = npzfile['b3']


    def ReLU(self, Z):
        #activation function
        return np.maximum(0, Z)
    

    def ReLU_derivative(self, Z):
        return np.where(Z<=0, 0, 1)


    def softmax(self, Z):
        #softmax activation function
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
        
    

    def forward_propagation(self, X):
        #applies weights and biases from each layer and then applies activation functions
        self.Z1 = self.W1.dot(X) + self.b1  #dot method carries out matrix multiplication
        self.A1 = self.ReLU(self.Z1)

        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.ReLU(self.Z2)

        self.Z3 = self.W3.dot(self.A2) + self.b3
        self.A3 = self.softmax(self.Z3)
    
    
    def one_hot(self, Y):
        # encode the labels into one-hot encoding (1300, 1) -> (1300, 2) as our output layer is 2 neurons
        # (0(ham) -> [1, 0], 1(spam) -> [0, 1] for each label) 
        #tranpose to get back into correct shape
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T


    def backward_propagation(self, X, Y):
        m = Y.size
        one_hot_Y = self.one_hot(Y)

        # Calculate gradients for each layer

        # dZ3 -> subtracts the desired output (one-hot encoded labels) from the actual output (softmax probabilities)
        # This difference represents the error for each element in the output vector
        
        #output layer gradients
        dZ3 = self.A3 - one_hot_Y
        dW3 = 1. / m * dZ3.dot(self.A2.T)                 #gradient of the loss with respect to W3 (chain rule)
        db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True) #gradient of the loss with respect to b3
        
        #hidden layer 2 gradients
        dA2 = self.W3.T.dot(dZ3)
        dZ2 = dA2 * self.ReLU_derivative(self.Z2)
        dW2 = 1. / m * dZ2.dot(self.A1.T)
        db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
        
        #hidden layer 1 gradients
        dA1 = self.W2.T.dot(dZ2)
        dZ1 = dA1 * self.ReLU_derivative(self.Z1)
        dW1 = 1. / m * dZ1.dot(X.T)
        db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
        
        #update weights and biases
        self.W3 = self.W3 - self.learning_rate * dW3
        self.b3 = self.b3 - self.learning_rate * db3
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1


    def gradient_descent(self, X, Y, X_test, Y_test, epochs):
        max_acc = (0, 0)
        loss_train_xy = []
        acc_test_xy = []
        for i in range(epochs):
            self.forward_propagation(X)
            self.backward_propagation(X, Y)
            loss_train_xy.append((i, self.compute_loss(Y)))
            self.predict(X_test)
            acc_test_xy.append((i, self.get_accuracy(Y_test)))
            if self.get_accuracy(Y_test) > max_acc[1]:
                max_acc = (i, self.get_accuracy(Y_test))
                #self.save_model("best_model_lol.npz")
            self.predict(X)
            if i % 100 == 0:
                print("Epoch: ", i)
                print("Accuracy: ", self.get_accuracy(Y))
                print("Loss: ", self.compute_loss(Y))

        return acc_test_xy, loss_train_xy, max_acc
        
    def compute_loss(self, Y): 
        #cross entropy loss
        m = Y.size
        one_hot_Y = self.one_hot(Y)
        log_probs = np.multiply(one_hot_Y, np.log(self.A3))
        loss = - 1. / m * np.sum(log_probs)
        return loss
    
    def get_accuracy(self, Y):
        predictions = np.argmax(self.A3, 0)
        return np.mean(predictions == Y)
    
    
    def train(self, X_train, Y_train, X_test, Y_test, epoches=1000, learning_rate=0.01):
        self.learning_rate = learning_rate
        acc_test_xy, loss_train_xy, max_acc = self.gradient_descent(X_train, Y_train, X_test,Y_test, epoches)
        return acc_test_xy, loss_train_xy, max_acc
    
        
    def predict(self, X):
        if X.shape[0] != 54:
            X = X.T
        self.forward_propagation(X)
        return np.argmax(self.A3, 0)
        

# 20, 20 800, 0.23 -> 0.94
# 20, 20 2200, 0.08 -> 0.95
# 20, 20 1000, 0.08 -> 0.955
# 20, 20 2900, 0.03 -> 0.955 frequently
# 20, 20 10000, 0.01 -> 0.96 #performs best on shuffled data, 12000 epcoch better

def plot():
    acc_test_xy = []
    train_loss_xy = []
    test_nn = SpamClassifierNN(hidden_size1=20, hidden_size2=20)
    acc_test_xy, train_loss_xy, max_acc = test_nn.train(X_train, Y_train, X_test, Y_test, epoches=15000, learning_rate=0.01)
    print(max_acc)

    # Unzip the (x, y) tuples
    acc_test_x, acc_test_y = zip(*acc_test_xy)
    train_loss_x, train_loss_y = zip(*train_loss_xy)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(acc_test_x, acc_test_y, label='Test Accuracy', color='orange')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.4, 1)  # set y-axis limits for accuracy

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(train_loss_x, train_loss_y, label='Train Loss', color=color)

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 0.8)  # set y-axis limits for loss

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Accuracy and Loss')
    fig.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix():
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    predictions = classifier.predict(X_test)
    for i in range(len(predictions)):
        if predictions[i] == 1 and Y_test[i] == 1:
            TP += 1
        elif predictions[i] == 1 and Y_test[i] == 0:
            FP += 1
        elif predictions[i] == 0 and Y_test[i] == 0:
            TN += 1
        elif predictions[i] == 0 and Y_test[i] == 1:
            FN += 1
    return TP, FP, TN, FN

def precision():
    TP, FP, TN, FN = confusion_matrix()
    return TP / (TP + FP)

def plot_confusion_matrix():
    TP, FP, TN, FN = confusion_matrix()
    confusion = np.array([[TP, FP], [FN, TN]])
    fig, ax = plt.subplots()
    im = ax.imshow(confusion, cmap=plt.cm.Blues)
    
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["Spam", "Ham"])
    ax.set_yticklabels(["Spam", "Ham"])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="red")

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Precision: {precision()})')
    plt.colorbar(im)
    plt.show()

plot_confusion_matrix()
TP, FP, TN, FN = confusion_matrix()
print("True Positives: ", TP)
print("False Positives: ", FP)
print("True Negatives: ", TN)
print("False Negatives: ", FN)


if __name__ == "__main__":
    classifier = SpamClassifierNN(hidden_size1=20, hidden_size2=20)
    acc_test_xy, train_loss_xy, max_acc = classifier.train(X_train, Y_train, X_test, Y_test, epoches=8200, learning_rate=0.01)
    print(max_acc)
    #classifier.load_model("best_model_test.npz")
    classifier.predict(X_test)
    print("Test accuracy: ", classifier.get_accuracy(Y_test))
    # plot()