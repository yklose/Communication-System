# pdc project


import numpy as np
import math
import operator

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
    w = [0]*(1000)              # example what is the time interval?
    codeword = codewords[0]     # per bit or whole message in once?
    
    # use root raised cosine functions!
    for i in range(num_bits):
        c = codeword[i]
        w_temp = [0]*(1000)
        for j in range(-500,500):
            t = T/500*j
            term = (1+beta)*(t-i)/T
            phi = 4*beta/(math.pi*math.sqrt(T))*(math.cos(term*math.pi)+(1-beta)*math.pi/(4*beta)*np.sinc(term))/(1-math.pow((4*beta*(t)/T),2))
            w_temp[j-1] = float(codeword[i])*phi
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


#to do: plot the recieved signal!
 
waveform_former()

    