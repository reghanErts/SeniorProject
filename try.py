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
    LEARNING_RATE = 0.55

    def __init__(self, no_of_in_nodes, no_of_out_nodes,
                 no_of_hidden_nodes, input_bias=None, hidden_layer_wghts=None,
                 hidden_layer_bias=None,out_layer_wghts=None, out_layer_bias=None,
                 ):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes

        self.input_layer = NeuronLayer(no_of_in_nodes, input_bias)
        self.hidden_layer = NeuronLayer(no_of_hidden_nodes, hidden_layer_bias)
        self.out_layer = NeuronLayer(no_of_out_nodes, out_layer_bias)
        self.init_wghts_from_inputs_to_hidden_layer_neurons(hidden_layer_wghts)
        self.init_wghts_from_hidden_layer_neurons_to_out_layer_neurons(out_layer_wghts)

    def init_wghts_from_inputs_to_hidden_layer_neurons(self, hidden_layer_wghts):
        wght_no = 0
        print("Im here *********************************")
        for inputs in range(len(self.input_layer.neurons)):
            for hidden in range(len(self.hidden_layer.neurons)):
                if not hidden_layer_wghts:
                    #sets the wght values:
                    #if none random
                    #if place it in the array at the neuron of whatever iteration we are on 
                    self.hidden_layer.neurons[hidden].wghts.append(random.random())
                else:
                    self.hidden_layer.neurons[hidden].wghts.append(hidden_layer_wghts[wght_no])
                wght_no += 1
            print('wght_no ', wght_no)

    def init_wghts_from_hidden_layer_neurons_to_out_layer_neurons(self, out_layer_wghts):
        wght_no = 0
        for output in range(len(self.out_layer.neurons)):
            for hidden in range(len(self.hidden_layer.neurons)):
                if not out_layer_wghts:
                    #if there is no weight make a random value between 1 and 0
                    self.out_layer.neurons[output].wghts.append(random.random())
                    #print("I made it ************************************")
                else:
                    # make sure im accessing the right spot in the array
                    self.out_layer.neurons[output].wghts.append(out_layer_wghts[wght_no])
                wght_no += 1
            #print('wght no ', wght_no)
        #print("wghts len at creation: ", len(out_layer_wghts))


    def fd_frwd(self, inputs):
        #print("inputs: ", inputs)
        hidden_layer_outs = self.hidden_layer.fd_frwd(inputs)
        #print("hidden layer outs: ", hidden_layer_outs)

        return self.out_layer.fd_frwd(hidden_layer_outs)
        #print('outlayer feed forward: ', self.out_layer.fd_frwd(hidden_layer_outs))

   #**********   TRAIN   *********************************************************************************

    def train(self, training_inputs, training_outs):
        #   "who" and "wih" are the index of wghts from hidden to out layer neurons and input to hidden layer neurons
        #print("Im here *********************************")
        self.fd_frwd(training_inputs)
        #print("fd_frwd(training_inputs): ",self.fd_frwd(training_inputs))
        #print("Im here *********************************")
        #out neurons deltas

        pd_errs_wrt_out_neuron_total_net_input = [0] * len(self.out_layer.neurons)
        #print("p: ",len(pd_errs_wrt_out_neuron_total_net_input[output]))
        #print("pnan: ", pd_errs_wrt_out_neuron_total_net_input)
        for output in range(len(self.out_layer.neurons)):

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
        for hidden in range(len(self.hidden_layer.neurons)):
            # calc the derivative of the err in respects to the out
            #of each hidden layer neuron

            d_err_wrt_hidden_neuron_out = 0

            for output in range(len(self.out_layer.neurons)):
                #for every node in the hidden layer
                #for every output neuron
                #multiply the output error times the weight between them
                d_err_wrt_hidden_neuron_out += pd_errs_wrt_out_neuron_total_net_input[
                    output] * self.out_layer.neurons[output].wghts[hidden]

            #pd E/ pd z at j = dE/dy at j * pd z at j/ pd
            #set the total err of this branch = the error of the output times the weight) * 
            # the value of the hidden layer neuron we ar elooking at
            pd_errs_wrt_hidden_neuron_total_net_input[hidden] = d_err_wrt_hidden_neuron_out * \
                self.hidden_layer.neurons[hidden].calc_pd_total_net_input_wrt_input()
        #update out neuron wghts
        #whooo = []
        for out in range(len(self.out_layer.neurons)):
            print("Im here *********************************")
            #who is the matrix of wghts between the hidden and out layers
            #for every output neuron we are looking at the weights connected
            # we take the output neuron we are looking at times
            # the weight we are looking at
            #now multiple tha times the the weight error - = the new weight
            for who in range(len(self.out_layer.neurons[out].wghts)):
                #pd E at j/ pd w at index i,j = pd E/pd z at j *
                    # pd z at j/ pd w at index i,j
                pd_err_wrt_wght = pd_errs_wrt_out_neuron_total_net_input[out] * \
                    self.out_layer.neurons[out].calc_pd_total_net_input_wrt_wght(
                        who)
                #print("pewontni", pd_errs_wrt_out_neuron_total_net_input[out])
                #print("out_layer.neuron[out].calc_pd_total_net_input_wrt_wght(who)",
                #      self.out_layer.neurons[out].calc_pd_total_net_input_wrt_wght(who))
                #print(" PD_ERR_WRT_wght", pd_err_wrt_wght)
                #Delta W = alpha * pd E at ju / pd w at i

                # new weight equals the (total error of output times the weight it has) * the learning rate  
                self.out_layer.neurons[out].wghts[who] -= self.LEARNING_RATE * \
                    pd_err_wrt_wght
                #for wght in self.out_layer.neurons:
                #print('\n\n\n*********')
                #print(self.out_layer.neurons[out].wghts[who])
                #print('**********\n\n\n')

                # print("whooo: ",self.out_layer.neurons[out].wghts[who].append(whooo))
                #print("out_layer.neuron[out].wghts:", self.out_layer.neurons[out].wghts[who])

        #update hidden wghts
        for hidden in range(len(self.hidden_layer.neurons)):
            #wih wghts in hidden layer
            for wih in range(len(self.hidden_layer.neurons[hidden].wghts)):
                #pd E at j / pd w at i = pd E/ pd z at j * pd z at j/ pd w at u
                pd_err_wrt_wght = pd_errs_wrt_hidden_neuron_total_net_input[hidden] * \
                    self.hidden_layer.neurons[hidden].calc_pd_total_net_input_wrt_wght(wih)
                #print("pd_error_wrt_wght: ", pd_err_wrt_wght[wih])
                #Delta = alpha * pd E at j/ pd w at i
                self.hidden_layer.neurons[hidden].wghts[wih] -= self.LEARNING_RATE * pd_err_wrt_wght

    def calc_total_err(self, training_sets):
        total_err = 0
        for t in range(len(training_sets)):
            training_inputs, training_outs = training_sets[t]
            #print("training_inputs: ",training_inputs)
            self.fd_frwd(training_inputs)
            #print("training_outs: ",training_outs)
            for out in range(len(training_outs)):
                #print("total_err ", total_err)
                total_err += self.out_layer.neurons[out].calc_err(
                    training_outs[out])
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
    def __init__(self, no_neurons, bias):

        # Every neuron in a layer shares the same bias
        #random comes up with a rand nober between 0 and 1
        self.bias = bias if bias else random.random()

        #print("bias hereee: " , bias)

        self.neurons = []
        #appends a refecence to the nueron class taking the bias and setting eveeryting in the neuron with the value
        #of the bias. appending those returning values to the specific neuron we are looking at
        # and stores it that that layer
        for i in range(no_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print('Neuron', n)
            for w in range(len(self.neurons[n].wghts)):
                print('wght:', self.neurons[n].wghts[w])
            print('  Bias:', self.bias)

    def fd_frwd(self, inputs):
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
        #each neuron get set its own wghts and bias
        self.bias = bias
        self.wghts = []
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
            print("length of wghts: ", len(self.wghts))
            print("value of input: ", self.inputs[i])
            print("value of wght: ", self.wghts[i])'''
            total += self.inputs[i] * self.wghts[i]
        #print("total = inputs * wghts: ", total)
        #print("bias: ",self.bias)
        #print("total+ bias: ", total+ self.bias)
        return total + self.bias

    # logistic function to squash the out of the neuron
    #activation function

    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected out
    #
    # Now we have the err with respect to the out and
    # the deriv of the out with respect to the total net input  we can calc
    # the pd err with respect to the total net input.
    #
    def calc_pd_err_wrt_total_net_input(self, target_out):
        return self.calc_pd_err_wrt_out(target_out) * self.calc_pd_total_net_input_wrt_input()

    # The err for each neuron is calcd by the Mean Square Error method:
   # https: // stackoverflow.com/questions/39064684/mean-squared-error-in-python?rq = 1
    def calc_err(self, target_out):
        #print("target - out: " ,target_out, self.out)
        #print("answer: ", 0.5 * (target_out - self.out) ** 2)
        return 0.5 * (target_out - self.out) ** 2

    
    #how much the error changes depending on the output
    def calc_pd_err_wrt_out(self, target_out):
        return -(target_out - self.out)

    # The total net input into the neuron is squashed using log func to calc the neuron's output

    # The deriv (not pd since there is only one var) of the output then is
    def calc_pd_total_net_input_wrt_input(self):
        return self.out * (1 - self.out)

    # The total net input is the wghted sum of all the inputs to the neuron and their respective wghts

    def calc_pd_total_net_input_wrt_wght(self, index):
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

    '''nn = NeuralNetwork(2, 2, 1, hidden_layer_wghts=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_wghts=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
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
nn = NeuralNetwork(len(big_data[0][0]), len(big_data[0][1]), 5 )
#print("big D:",len(big_data[0][0]))
#print("big D2:", len(big_data[0][1]))
for j in range(50):
    for i in range(len(big_data)-900):

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
        #nn.fd_frwd(training_inputs)
        #nn.train(training_inputs, training_outs)
        #nn.inspect()

        print("hello: ", nn.train(training_inputs, training_outs))
        print("i ,  err: ", i, nn.calc_total_err(big_data))
        #print("evaluated outs: ", nn.fd_frwd(training_inputs))

        '''for b in range(len(big_data[i][1])):
            print("B is ", b)
            print("len:", len(big_data[i][1]))'''

        #fir every instcance if the perdiction is correct add it to a count and at the end divide by total
        # CHNAGE TH
        # E TRAINING IN AND TRAING OUT TO WORK WITH JUST THE INSTANCES
        outs = nn.fd_frwd(training_inputs)
        print("outs:", outs)
        #print("training-outs: ", nn.fd_frwd(training_inputs))
        if outs[0] > .5:
            if training_outs[0] == 1.0:
                print("you got a 1 : ", you_got_one)
                you_got_one += 1
            else:
                print("wrong")
                print("outs *******************************", training_outs)
        elif outs[0] < .49:
            if training_outs[0] == 0:
                print("you got a 0: ", you_got_zero)
                you_got_zero += 1
            else:
                print("you go thtis wrong")


count = you_got_one + you_got_zero
print("count: ", count)
print("length: ", len(big_data))
calc_accuracy = (count / (48*50))
average = calc_accuracy * 100
print("accuracy: ", average, "%")

#print("len:", len(big_data[i][1]))
for val in range(len(big_data_validate)):
    training_inputs_val = big_data_validate[val][0]
    training_outs_val = big_data_validate[val][1]
    #training_inputs_val, training_outs_val = random.choice(big_data)
    outs_val = nn.fd_frwd(training_inputs_val)
    print("nn:", outs_val)
    print("value, in :", val, training_inputs_val)
    print("tr val: ", training_outs_val)
    if outs_val[0] > .5:
        if training_outs[0] == 1.0:
            print("you got a 1 : ", you_got_one)
            you_got_one += 1
        else:
            print("wrong")
            print("outs *******************************", training_outs)
    elif outs_val[0] < .49:
        if training_outs[0] == 0:
            print("you got a 0: ", you_got_zero)
            you_got_zero += 1
        else:
            print("you go thtis wrong")

    #print("touXXXXXXhv;lvgkjvgf'[cf]:", training_outs)
