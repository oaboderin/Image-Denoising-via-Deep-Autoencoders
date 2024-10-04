import numpy
import os
import math
from random import random
from math import exp
import matplotlib.pyplot



#Functions:
def initialize_weights(n_inputs, n_hidden, n_outputs, n_layer):
    weights = list()
    for n in range(n_layer):
        n_inputs = n_hidden
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        n_inputs = n_hidden
        weights.append(hidden_layer)
		#n_inputs = len(hidden_layer)
	    
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
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
        network_error = []
        for m in range(n_times):
                error_display = 0
                v = 0

               

                #learning_rate = 0.98**(m)
                #learning_rate = numpy.power(learning_rate, (m-1))
                for row in x:
                        outputs = forward_propagate(weights, row)
                        #print('row', row, 'k', m, 'out', outputs)
                        expected = y[v]
                        error_display += sum([(outputs[f]-expected[f])**2 for f in range(n_outputs)])
                        backward_propagate(weights, expected)
                        v = v+1
                        update_weights(weights, row, learning_rate)
            
                #print(weights)
                #print("iteration", k, "error", error_display)
                #print(m, "error", error_display/400)
                network_error.append(error_display/400)

        #matplotlib.pyplot.figure()
        #matplotlib.pyplot.plot(network_error)
        #matplotlib.pyplot.title("Iteration Number vs Error")
        #matplotlib.pyplot.xlabel("Iteration Number")
        #matplotlib.pyplot.ylabel("Error")
        #matplotlib.pyplot.show()
        #print(network_error)
        return network_error

def neural_network3(weights, x, y, learning_rate, n_times, n_outputs):
#performs all the operations: forward propagates, calcuates derivatives and 
#updates weights
        network_error = []
        for m in range(n_times):
                error_display = 0
                v = 0
                learning_rate = 0.98**(m)
                #learning_rate = numpy.power(learning_rate, (m-1))
                for row in x:
                        outputs = forward_propagate(weights, row)
                        #print('row', row, 'k', m, 'out', outputs)
                        expected = y[v]
                        error_display += sum([(outputs[f]-expected[f])**2 for f in range(n_outputs)])
                        backward_propagate(weights, expected)
                        v = v+1
                        update_weights(weights, row, learning_rate)
            
                #print(weights)
                #print("iteration", k, "error", error_display)
                #print(m, "error", error_display/400)
                network_error.append(error_display/400)
        return network_error


fname = os.path.join("Data.txt")
data = numpy.loadtxt(fname, delimiter=",")
for i in range(400):

 if data[i,2]<0:
     data[i,2]=0
 if data[i,3]<0:
     data[i,3]=0

rows, cols = (400, 2)
x = [[0.1*numpy.random.randn() for i in range(cols)] for j in range(rows)]
y = [[0.1*numpy.random.randn() for i in range(cols)] for j in range(rows)]
#print(y[0,1])
print(numpy.size(x))
print(numpy.size(y))

     
for k in range(400):
    x[k] = [data[k,0], data[k,1]]
    y[k] = [data[k,2], data[k,3]] # for j in range(2)]

#print(y[0])
#for i in range(50):
    #print("i2", data[i,2], "i3", data[i,3],"y", y[i])
    #print("i0", data[i,0], "i1", data[i,1],"x", x[i])



learning_rate_init = 0.01
#Data from dataset
n_inputs = 2
n_outputs = 2
n_hidden = 2 #output and hidden layer
n_layer = 1
n_times = 3000 #set number of iterations/epochs
weights = initialize_weights(n_inputs, n_hidden, n_outputs, n_layer)
#weights = initialize_weights(n_inputs, n_hidden, n_outputs)
                            
#for layer in weights:
    #print("1",layer)
    #new_inputs =[]
    #print("2",new_inputs)
    #for z in layer:
    #   print("3",z)
network1 = neural_network(weights, x, y, learning_rate_init, n_times, n_outputs)
weights2 = initialize_weights(n_inputs, n_hidden, n_outputs, 2)
network2 = neural_network(weights2, x, y, 0.01, n_times, n_outputs)
weights3 = initialize_weights(n_inputs, n_hidden, n_outputs, 4)
network3 = neural_network(weights3, x, y, 0.1, n_times, n_outputs)

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(network1, label = 'One hidden layer')
matplotlib.pyplot.plot(network2, label = 'Two hidden layers')
matplotlib.pyplot.plot(network3, label = 'Four hidden layers')
matplotlib.pyplot.legend()
matplotlib.pyplot.title("Error vs Number of Iterations")
matplotlib.pyplot.xlabel("Number of Iterations")
matplotlib.pyplot.ylabel("Error")
matplotlib.pyplot.show()




                


