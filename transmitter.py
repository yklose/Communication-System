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
    text_array = list(f.read())
    codeword_str = ""
    n = len(text_array)
    for i in range(n):
        # convert text to ascii int 
        ascii_int = ord(text_array[i])
        # convert int to 8 bit
        ascii_bits = '{0:08b}'.format(ascii_int)
        
        codeword_str = codeword_str + " " + str(ascii_bits) + " "
    
    print(codeword_str)
    return codeword_str
        

# waveform former creates signal based on codewords
# codewords are send as one large array (concatenade)

def waveform_former():
    
    # create codewords
    codewords = create_codewords(datapath)
    codeword = codewords.replace(" ", "")
    
    # idea: modify codewords
    # one could modify the codewords such that 0 is replaced with -1
    # maybe better for error probablitiy
    
    # set parameters
    beta = 1/2
    f_sample = 22050
    T = 1/f_sample
    num_s = int(200)                        # change if needed
    num_s_h = int(num_s/2)
      
    
    #codeword = [1,1,1,0,0]                  # just for testing! 
    num_bits = len(codeword)                # number of bits transmitted
    w = [0]*((num_bits+1)*num_s)            # create w
    
    # use root raised cosine functions!
    for i in range(num_bits):
        c = codeword[i]
        w_temp = [0]*((num_bits+1)*num_s)
        for j in range(-num_s_h-num_s*(i),(((num_bits+1)*num_s)-num_s_h)-i*num_s):
            t = T/num_s_h*j
            
            # implementation of root raised cosine function
            term_plus = (1+beta)*(t)/T
            term_minus = (1-beta)*(t)/T
            sinc_term = np.sinc(term_minus)
            denomitor = 1-math.pow((4*beta*(t)/T),2)
            if denomitor == 0:
                
                print((4*beta*(t)/T))
            
                
            phi = 4*beta/(math.pi*math.sqrt(T))*(math.cos(term_plus*math.pi)+(1-beta)*math.pi/(4*beta)*sinc_term)/(denomitor)
            
            # w_temp is for one codeword only
            w_temp[(j+num_s_h+num_s*i)] = float(codeword[i])*phi
            
        
        # add the w_temp to the signal (w = w + w_temp)
        print("Step: " + str(i+1) + "/" + str(num_bits))
        w = list(map(operator.add, w,w_temp))
    
    # within certain frequency in order to go through the filter
    # TODO: write function that transforms signal to right frequency for passband filter
    
    # convert signal to txt file for output
    signal = ""
    for i in range(len(w)):
        signal = signal + str(w[i]) + "\n"
    signal_file = open("signal.txt", "w")
    signal_file.write(signal)
    signal_file.close()
    
    plot_channel_output.plot_output("signal.txt")

#to do: plot the recieved signal!
 
waveform_former()

    