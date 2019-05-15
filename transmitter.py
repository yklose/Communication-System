# pdc project


import numpy as np
import math
import operator
import plot_channel_output

datapath = "text.txt"

# reads a text input file and returns its codewords. 
# using a 8 bit translation (Ascii - Code)

def create_codewords(datapath):
    # convert to codewords
    f = open(datapath, 'r')
    text_array = f.read().split(' ')
    codeword_str = ""
    n = len(text_array)
    for i in range(n):
        # convert text to ascii int 
        ascii_int = ord(text_array[i])
        # convert int to 8 bit
        ascii_bits = '{0:08b}'.format(ascii_int)
        
        codeword_str = codeword_str + " " + str(ascii_bits) + " "
    
    #print(codeword_str)
    
    return codeword_str
    # save the file
    
    
# waveform former!

def waveform_former():
    
    # create codewords
    codewords = create_codewords(datapath)
    codewords = codewords.split()
    
    # set parameters
    beta = 1/2
    f_sample = 22050
    T = 1/f_sample
    
    num_bits = 8
    w = [0]*(8000)              # example what is the time interval?
    codeword = codewords[0]     # per bit or whole message in once?
    
    codeword = [0,0,1,0,0,0,0,0]
    # use root raised cosine functions!
    for i in range(num_bits):
        c = codeword[i]
        print("C: " + str(c))
        w_temp = [0]*(8000)
        for j in range(-500,500):
            t = T/500*j
            term_plus = (1+beta)*(t)/T
            term_minus = (1-beta)*(t)/T
            phi = 4*beta/(math.pi*math.sqrt(T))*(math.cos(term_plus*math.pi)+(1-beta)*math.pi/(4*beta)*np.sinc(term_minus))/(1-math.pow((4*beta*(t)/T),2))
            #print(phi)
            w_temp[i*500+(j+500)] = float(codeword[i])*phi
        w = list(map(operator.add, w,w_temp))
    
    # within certain frequency in order to go through the filter
    
    # convert signal to txt file for output
    signal = ""
    for i in range(len(w)):
        signal = signal + str(w[i]) + "\n"
    #print(signal)
    signal_file = open("signal.txt", "w")
    signal_file.write(signal)
    signal_file.close()
    
    plot_channel_output.plot_output("signal.txt")

#to do: plot the recieved signal!
 
waveform_former()

    