# plot of output

import matplotlib.pyplot as plt
import numpy as np

datapath1 = "output.txt"
datapath2 = "signal.txt"

def plot_output(datapath):
    f = open(datapath, 'r')
    array = f.read().split('\n')
    length = len(array)
    plot_array = array[:(length-1)]
    max_index = np.argmax(plot_array)
    #print("Maximum at: " + str(max_index/length))
    plot_array = list(map(float, plot_array))
    plt.plot(plot_array)
    plt.show()
    #print(array)
    
#plot_output(datapath1)   
    