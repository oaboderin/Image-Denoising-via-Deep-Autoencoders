from keras.datasets import mnist
import numpy
from matplotlib import pyplot
import os
import math
from random import random
from math import exp

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(str(x_train.shape))
#print(str(x_test.shape))
xtrainx = x_train.reshape(60000,784)

train_x = numpy.divide(xtrainx, 255.0)
#print(numpy.size(train_x[60]))

rows, cols = (15, 784)
data = [[random() for o in range(cols)] for u in range(rows)]
#y = [[0.1*numpy.random.randn() for i in range(cols)] for j in range(rows)]
#print(y[0,1])


     
for k in range(15):
    data[k] = train_x[k]

#print(data)

#print(xtrainx)
#xtrain = random((100, 28, 28))
#for i in range(100):
#    xtrain[i] = x_train[i]
#print(xtrain.shape)
#print((xtrain[0]/255))
#train_x = xtrain.astype('float32')/255.0


#Functions:
def initialize_weights(n_inputs, n_hidden, n_outputs, n_layer, leninnerlayer, lenouterlayer):
    weights = list()
    for n in range(n_layer):
        if n == 0 or n == 3:
            n_hidden = lenouterlayer
            #hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
            #n_hidden = n_inputs
        if n == 1 or n == 2:
            n_hidden = leninnerlayer
        hidden_layer = [{'weights':[0.1*numpy.random.rand() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        n_inputs = n_hidden
        weights.append(hidden_layer)
		#n_inputs = len(hidden_layer)
	    
    output_layer = [{'weights':[0.1*numpy.random.rand() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    weights.append(output_layer)
    return weights
    
def sop(weights, inputs):
        sop = weights[-1] #to add the bias
        for c in range(len(weights)-1):
                sop += weights[c] * inputs[c]
        return sop
 
def sigmoid(sop):
    #return numpy.power((1 + numpy.exp(-sop)),-1)
        return 1.0 / (1.0 + exp(-sop))

def forward_propagate(weights, row):
        inputs = row
        for layer in weights:
                new_inputs = []
                for neuron in layer: #run for loop 
                        a = sop(neuron['weights'], inputs)
                        neuron['output'] = sigmoid(a)
                        new_inputs.append(neuron['output'])
                inputs = new_inputs
        return inputs


def sigmoid_sop_deriv(output):
        return (output) * (1.0 - output)

                        
def backward_propagate(weights, expected): 
        for e in reversed(range(len(weights))): 
                layer = weights[e]
                derivatives = list()
                if e != len(weights)-1: 
                        for j in range(len(layer)):
                                error = 0.0
                                for neuron in weights[e + 1]:
                                        error += (neuron['weights'][j] * neuron['delta'])
                                        #print("neuron in j", neuron)
                                derivatives.append(error)
               
                else:
                        for w in range(len(layer)):
                                neuron = layer[w]
                                derivatives.append(neuron['output'] - expected[w]) #output layer 
               
                for h in range(len(layer)): 
                        neuron = layer[h]
                        neuron['delta'] = derivatives[h] * sigmoid_sop_deriv(neuron['output'])
                        #print("neuron in h", neuron)

            
def update_weights(weights, row, learning_rate):
        for b in range(len(weights)):
                inputs = row
                
                if b != 0:
                        inputs = [neuron['output'] for neuron in weights[b - 1]]
                        
                for neuron in weights[b]:
                        #print("x", row, "inputs", inputs)
                        for g in range(len(inputs)):
                                neuron['weights'][g] -= learning_rate * neuron['delta'] * inputs[g] #chain rule still being applied for 
                        neuron['weights'][-1] -= learning_rate * neuron['delta'] #bias
                        


def neural_network(weights, x, y, learning_rate, n_times, n_outputs):
#performs all the operations: forward propagates, calcuates derivatives and 
#updates weights
        for m in range(n_times):
                error_display = 0
                v = 0
                learning_rate = 0.98**((m+1)-1)
               
                
                #learning_rate *= learning_rate
                for row in x:
                        outputs = forward_propagate(weights, row)
                        #print('row', row, 'k', m, 'out', outputs)
                        expected = y[v]
                        error_display += sum([(outputs[f]-expected[f])**2 for f in range(n_outputs)])
                        backward_propagate(weights, expected)
                        v = v+1
                        update_weights(weights, row, learning_rate)
                final_error = error_display/15
                        
            
                #print(weights)
                #print("iteration", k, "error", error_display)
                if m == n_times - 1:
                    print(m, "error", error_display/15, "rate", learning_rate)
        return final_error
                



#data = train_x.reshape((100,784))  #xtrain for now

#print(data[60])

learning_rate = 0.01 #start at 0.01 
#Data from dataset
n_inputs =784 #change
n_outputs = 784 #change
n_hidden = 256
n_layer = 4
n_times = 1500 #set number of iterations/epochs
weights1 = initialize_weights(n_inputs, n_hidden, n_outputs, n_layer, 32, 256)
#for layer in weights:
#	print(layer)
neural1 = neural_network(weights1, data, data, learning_rate, n_times, n_outputs)

weights2 = initialize_weights(n_inputs, n_hidden, n_outputs, n_layer, 64, 256)
weights3 = initialize_weights(n_inputs, n_hidden, n_outputs, n_layer, 128, 256)
weights4 = initialize_weights(n_inputs, n_hidden, n_outputs, n_layer, 256, 600)
weights5 = initialize_weights(n_inputs, n_hidden, n_outputs, n_layer, 512, 600)

neural2 = neural_network(weights2, data, data, learning_rate, n_times, n_outputs)
neural3 = neural_network(weights3, data, data, learning_rate, n_times, n_outputs)
neural4 = neural_network(weights4, data, data, learning_rate, n_times, n_outputs)
neural5 = neural_network(weights5, data, data, learning_rate, n_times, n_outputs)
#xtrain, first 500 images
#normalize the array (divide by 255)
#normalize error calculations
#a = numpy.arange(75).reshape((3,5,5))
#b = numpy.zeros(a.shape)
#for i in range(3):
    #b[i] = a[i]

#print(b)

#for i in range(9):  
#pyplot.subplot(330 + 1 + 0)
#pyplot.imshow(x_train[0], cmap=pyplot.get_cmap('gray'))
#pyplot.show()
                


# x-coordinates of left sides of bars
left = [1, 2, 3, 4, 5]

# heights of bars
height = [neural1, neural2, neural3, neural4, neural5]

# labels for bars
tick_label = ['32', '64', '128', '250', '512']

# plotting a bar chart
matplotlib.pyplot.bar(left, height, tick_label = tick_label,
		width = 0.8, color = ['red', 'green','blue','yellow','orange'])

# naming the x-axis
matplotlib.pyplot.xlabel('Number of Neurons in 2nd Layer')
# naming the y-axis
matplotlib.pyplot.ylabel('Error')
# plot title
matplotlib.pyplot.title('Mnist Images Autoencoder')

# function to show the plot
matplotlib.pyplot.show()

