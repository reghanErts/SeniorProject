import numpy as np
import random
import math
from scipy.stats import truncnorm
import pandas as pd
import csv

#data frame
#for each line split on commas 
#put into the array

class NeuralNetwork:
    #might need to be ajusted depending on the steps
    LEARNING_RATE = 0.5
    def __init__(self,
                 no_of_in_nodes,no_of_out_nodes,
                 no_of_hidden_nodes, input_bias= None, hidden_layer_weights = None, 
                 hidden_layer_bias = None,
                 out_layer_weights=None, out_layer_bias=None,
                 ):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
    
        self.input_layer= NeuronLayer(no_of_in_nodes,input_bias)
        self.hidden_layer = NeuronLayer(no_of_hidden_nodes, hidden_layer_bias)
        self.out_layer = NeuronLayer(no_of_out_nodes, out_layer_bias)
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_out_layer_neurons(out_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_no = 0
        print("Im here *********************************")
        for inputs in range(len(self.input_layer.neurons)):
            for hidden in range(len(self.hidden_layer.neurons)):
                if not hidden_layer_weights:
                    #sets the weight values:
                    #if none random
                    #if place it in the array at the whatever iteration we are on
                    self.hidden_layer.neurons[hidden].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[hidden].weights.append(hidden_layer_weights[weight_no])
                weight_no += 1
            print('weight_no ',weight_no)

    def init_weights_from_hidden_layer_neurons_to_out_layer_neurons(self, out_layer_weights):
        weight_num = 0
        for output in range(len(self.out_layer.neurons)):
            for hidden in range(len(self.hidden_layer.neurons)):
                if not out_layer_weights:
                    self.out_layer.neurons[output].weights.append(random.random())
                    print("I made it ************************************")
                else:
    # make sure im accessing the right spot in the array 
                    self.out_layer.neurons[output].weights.append(out_layer_weights[weight_num])
                weight_num += 1
            print('weight num ',weight_num)
        #print("weights len at creation: ", len(out_layer_weights))
    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.no_of_in_nodes))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.out_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        #print("inputs: ", inputs)
        hidden_layer_outs = self.hidden_layer.feed_forward(inputs)
        #print("hidden layer outs: ", hidden_layer_outs)
        
        return self.out_layer.feed_forward(hidden_layer_outs)
        #print('outlayer feed forward: ', self.out_layer.feed_forward(hidden_layer_outs))

    
   #**********   TRAIN   *********************************************************************************

    def train(self, training_inputs, training_outs):  
        #   "who" and "wih" are the index of weights from hidden to out layer neurons and input to hidden layer neurons 
        #print("Im here *********************************")
        self.feed_forward(training_inputs)
        #print("feed_forward(training_inputs): ",self.feed_forward(training_inputs))
        #print("Im here *********************************")
        #out neurons deltas
        
        pd_errs_wrt_out_neuron_total_net_input = [0] * len(self.out_layer.neurons)
        #print("p: ",len(pd_errs_wrt_out_neuron_total_net_input[output]))
        #print("pnan: ", pd_errs_wrt_out_neuron_total_net_input)
        for output in range(len(self.out_layer.neurons)):

            #pd E/ pd z at j
            #print("p: ", len(pd_errs_wrt_out_neuron_total_net_input))
            #print("p.1: ", pd_errs_wrt_out_neuron_total_net_input[output])
            '''print("p2: ", len(training_outs))
            print("p2.1: ", training_outs[output])
            print("p3: ",(len(self.out_layer.neurons)))'''
           # print("p3.1: ", self.out_layer.neurons[output])
            #print("p3: ", len(calc_pd_err_wrt_total_net_input(training_outs)))
            pd_errs_wrt_out_neuron_total_net_input[output] = self.out_layer.neurons[output].calc_pd_err_wrt_total_net_input(
                training_outs[output])
            #print("what im looking for: ",pd_errs_wrt_out_neuron_total_net_input[output])
            
        #move to hidden neuron deltas
        pd_errs_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for hidden in range( len(self.hidden_layer.neurons)):
            # calc the derivative of the err in respects to the out 
                #of each hidden layer neuron
                #dE/dy at i = summation pd E/ pd z at i * pd z/pd y at j =
                # summation pd E/ pd z at j * w(weight)at index i,j
            d_err_wrt_hidden_neuron_out = 0

            for output in range(len(self.out_layer.neurons)):
                d_err_wrt_hidden_neuron_out += pd_errs_wrt_out_neuron_total_net_input[
                    output] * self.out_layer.neurons[output].weights[hidden]
            
            #pd E/ pd z at j = dE/dy at j * pd z at j/ pd
            pd_errs_wrt_hidden_neuron_total_net_input[hidden] = d_err_wrt_hidden_neuron_out * self.hidden_layer.neurons[hidden].calc_pd_total_net_input_wrt_input()
        #update out neuron weights
        #whooo = []
        for out in range (len(self.out_layer.neurons)):
            print("Im here *********************************")
            #who is the matrix of weights between the hidden and out layers
            for who in range (len(self.out_layer.neurons[out].weights)):
                #pd E at j/ pd w at index i,j = pd E/pd z at j * 
                    # pd z at j/ pd w at index i,j
                pd_err_wrt_weight = pd_errs_wrt_out_neuron_total_net_input[out] * self.out_layer.neurons[out].calc_pd_total_net_input_wrt_weight(who)
                #print("pewontni", pd_errs_wrt_out_neuron_total_net_input[out])
                #print("out_layer.neuron[out].calc_pd_total_net_input_wrt_weight(who)",
                #      self.out_layer.neurons[out].calc_pd_total_net_input_wrt_weight(who))
                #print(" PD_ERR_WRT_WEIGHT", pd_err_wrt_weight)
                #Delta W = alpha * pd E at ju / pd w at i
                self.out_layer.neurons[out].weights[who] -= self.LEARNING_RATE *pd_err_wrt_weight
                #for weight in self.out_layer.neurons:
                    #print('\n\n\n*********')
                    #print(self.out_layer.neurons[out].weights[who])
                    #print('**********\n\n\n')

                # print("whooo: ",self.out_layer.neurons[out].weights[who].append(whooo))
                #print("out_layer.neuron[out].weights:", self.out_layer.neurons[out].weights[who])

        #update hidden weights
        for hidden in range(len(self.hidden_layer.neurons)):
            #wih weights in hidden layer
            for wih in range(len(self.hidden_layer.neurons[hidden].weights)):
                #pd E at j / pd w at i = pd E/ pd z at j * pd z at j/ pd w at u
                pd_err_wrt_weight = pd_errs_wrt_hidden_neuron_total_net_input[hidden] * self.hidden_layer.neurons[hidden].calc_pd_total_net_input_wrt_weight(wih)
                #print("pd_error_wrt_weight: ", pd_err_wrt_weight[wih])
                #Delta = alpha * pd E at j/ pd w at i
                self.hidden_layer.neurons[hidden].weights[wih] -= self.LEARNING_RATE *  pd_err_wrt_weight

    def calc_total_err(self, training_sets):
        total_err = 0
        for t in range(len(training_sets)):
            training_inputs, training_outs = training_sets[t]
            #print("training_inputs: ",training_inputs)
            self.feed_forward(training_inputs)
            #print("training_outs: ",training_outs)
            for out in range(len(training_outs)):
                #print("total_err ", total_err)
                total_err += self.out_layer.neurons[out].calc_err(training_outs[out])
        print("end total Error:", total_err)
        return total_err
        
        
    def save(self):
        np.save('saved_wih.npy', self.wih)
        np.save('saved_who.npy', self.who)

    def load(self):
        self.wih = np.load('saved_wih.npy')
        self.who = np.load('saved_who.npy')
    def run(self, input_vector):
       """
      
"""


#*****************************************************************************************
class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        #random comes up with a rand number between 0 and 1
        self.bias = bias if bias else random.random()

        #print("bias hereee: " , bias)

        self.neurons = []
        #appends a refecence to the nueron class taking the bias and setting eveeryting in the neuron with the value 
        #of the bias. appending those returning values to the specific neuron we are looking at 
        # and stores it that that layer
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))


    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print('Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outs = []
        for neuron in self.neurons:
            outs.append(neuron.calc_out(inputs))
        
        #print("outs feedforward1: " , outs)
        return outs

    def get_outs(self):
        outs = []
        for neuron in self.neurons:
            outs.append(neuron.out)
            #print("outs get_outs1: ", outs[neuron])
        return outs
        

#******************************************************************************************
class Neuron:
    def __init__(self, bias):
        #each neuron get set its own weights and bias
        self.bias = bias
        self.weights= []
        #print("bias here: ", self.bias)
        

    def calc_out(self, inputs):
        self.inputs = inputs
        #print("inputs: ", inputs)
        self.out = self.squash(self.calc_total_net_input())
        #print("squash outs: ", self.out)
        return self.out

    def calc_total_net_input(self):
        total = 0
        #print("bias here: ", self.bias)
        for i in range(len(self.inputs)):
            #throws list index out of range err
            '''print ("length of inputs: ", len(self.inputs))
            print("length of weights: ", len(self.weights))
            print("value of input: ", self.inputs[i])
            print("value of weight: ", self.weights[i])'''
            total += self.inputs[i] * self.weights[i]
        #print("total = inputs * weights: ", total)
        #print("bias: ",self.bias)
        #print("total+ bias: ", total+ self.bias)
        return total + self.bias
    

    # logistic function to squash the out of the neuron
    #activation function
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected out
    #
    # Now we have the err wrt the out (∂E/∂yⱼ) and
    # the deriv of the out wrt the total net input (dyⱼ/dzⱼ) we can calc
    # the pd err with respect to the total net input.
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calc_pd_err_wrt_total_net_input(self, target_out):
        return self.calc_pd_err_wrt_out(target_out) * self.calc_pd_total_net_input_wrt_input()

    # The err for each neuron is calcd by the Mean Square Error method:
   # https: // stackoverflow.com/questions/39064684/mean-squared-error-in-python?rq = 1
    def calc_err(self, target_out):
        #print("target - out: " ,target_out, self.out)
        #print("answer: ", 0.5 * (target_out - self.out) ** 2)
        return 0.5 * (target_out - self.out) ** 2

    # The pd of the err wrt actual out then is calcd by:
    # = 2 * 0.5 * (target out - actual out) ^ (2 - 1) * -1
    # = -(target out - actual out)
   
    # Note that the actual out of the out neuron is often written as yⱼ and target out as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calc_pd_err_wrt_out(self, target_out):
        return -(target_out - self.out)

    # The total net input into the neuron is squashed using log func to calc the neuron's out:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # N: where j represents the out of the neurons in the layer we're looking at and i represents the layer below it
    
    # The deriv (not pd since there is only one var) of the out then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calc_pd_total_net_input_wrt_input(self):
        return self.out * (1 - self.out)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    
    # The pd of the total net input wrt a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ

    def calc_pd_total_net_input_wrt_weight(self, index):
        #print("index" , index)
        return self.inputs[index]

###
#**********************************************************************
#data processing function
# 2d to a aplit 2d array in function call
def first_array(small_array):
    deep_process = []
    attributes = []
    target = []
    for i in range(len(small_array)):
        if i < (3):
            #(len(small_array)-1)
            attributes.append(small_array[i])
        else:
            target.append(small_array[i])
    deep_process.append(attributes)
    deep_process.append(target)
    return deep_process

#********************************************************************************************************
if __name__ == "__main__":
    '''simple_network = NeuralNetwork(
        no_of_in_nodes=4, no_of_out_nodes=2,
        no_of_hidden_nodes=3, hidden_layer_weights=None,
        hidden_layer_bias=None,
        output_layer_weights=None, output_layer_bias=None,)
    simple_network.run([(3, 4)])'''
  #********************************data handling/ formatting*************************************
    #Read Necessary files
    #putting our train csv file into a 2d array
    readerTrain = np.loadtxt(
        #had to get rid of major to be able to use the other values as floats, skips row one (headers)
        open("dataset/train_reghanData_noMajor.csv", "r"), delimiter=",", skiprows=1)
        #open("train_reghanData_noMajor.csv", "r"), delimiter=",", skiprows=1)
    x = list(readerTrain)
    resultTrain = np.array(x).astype("float")
    #print ( 'train','\n', resultTrain)
    readerTest = np.loadtxt(
        #had to get rid of major to be able to use the other values as floats
        open("dataset/test_reghanData_noMajor.csv", "r"), delimiter=",", skiprows=1)
        #open("test_reghanData_noMajor.csv", "r"), delimiter=",", skiprows=1)
    x = list(readerTest)
    resultTest = np.array(x).astype("float")
    #print('test','\n',resultTest)

    # calls the function on the 2d array 
    #passes in the 2d that splits it and for 
    # every iteration i appends it to a new array 
    # ending up with a 3 d array
    big_data = []
    for a in range(len(resultTrain)):
        big_data.append(first_array(resultTrain[a]))    
    #print(big_data)

    big_data_validate = []
    for b in range(len(resultTest)):
        big_data_validate.append(first_array(resultTest[b]))
    #print(big_data_validate)
 
    '''nn = NeuralNetwork(2, 2, 1, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
    for i in range(10000):
        nn.train([0.05, 0.1], [0.01, 0.99])
        print(i, round(nn.calc_total_err([[[0.05, 0.1], [0.01, 0.99]]]), 9))

'''

# XOR example:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]
test = []
you_got_one = 0
you_got_zero = 0
count = 0
nn = NeuralNetwork(len(big_data[0][0]), len(big_data[0][1]), 4)
#print("big D:",len(big_data[0][0]))
#print("big D2:", len(big_data[0][1]))
for i in range(len(big_data)-500):
    
    print("hello**************************************************************************************")
    training_inputs = big_data[i][0]
    training_outs = big_data[i][1]

#training_inputs, training_outs = random.choice(big_data)

    '''training_test_inputs = [23.3729305949,3.48457837330, 32.456765]
    training_test_output= [1]
    nn.train(training_test_inputs, training_test_output)
    
    '''
    print("tin:", training_inputs)
    print("training-outputs value: ", training_outs)
    
    #print("touXXXXXXhv;lvgkjvgf'[cf]:", training_outs)
    #nn.feed_forward(training_inputs)
    #nn.train(training_inputs, training_outs)
    #nn.inspect()

    print("hello: ", nn.train(training_inputs, training_outs))
    print("i ,  err: ", i, nn.calc_total_err(big_data))
    #print("evaluated outs: ", nn.feed_forward(training_inputs))
    
    '''for b in range(len(big_data[i][1])):
        print("B is ", b)
        print("len:", len(big_data[i][1]))'''
        
        
    #fir every instcance if the perdiction is correct add it to a count and at the end divide by total 
    # CHNAGE TH
    # E TRAINING IN AND TRAING OUT TO WORK WITH JUST THE INSTANCES
    outs= nn.feed_forward(training_inputs)
    print("outs:", outs)
    #print("training-outs: ", nn.feed_forward(training_inputs))
    if outs[0] > .5 :
        if  training_outs[0] == 1.0:
            print("you got a 1 : ", you_got_one)
            you_got_one += 1
        else:
            print ("wrong")
            print("outs *******************************", training_outs)    
    elif outs[0] < .49 :
        if training_outs[0] == 0:
            print("you got a 0: ", you_got_zero)
            you_got_zero += 1
        else:
            print( "you go thtis wrong") 
                
            
count = you_got_one + you_got_zero
print("count: ", count)
print("length: ",len(big_data))
calc_accuracy = (count / 448)
average = calc_accuracy * 100
print("accuracy: ",average, "%" )    
                
#print("len:", len(big_data[i][1]))
'''for val in range(len(big_data_validate)):
    training_inputs_val = big_data_validate[val][0]
    training_outs_val = big_data_validate[val][1]
    #training_inputs_val, training_outs_val = random.choice(big_data)
    print ("nn:" , nn.feed_forward(training_inputs_val))
    print("value, in :", val, training_inputs_val)
    print("tr val: ", training_outs_val)'''
        
    #print("touXXXXXXhv;lvgkjvgf'[cf]:", training_outs)
