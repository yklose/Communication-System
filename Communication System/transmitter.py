# pdc project


import numpy as np
import math
import operator
from scipy.fftpack import fft, ifft, fftshift
import time


sampling_rate = int(102)

def run():
    datapath = "text.txt"
    create_codewords(datapath)
    waveform_former()
    #test()

# reads a text input file and returns its codewords. 
# using a 8 bit translation (Ascii - Code)

def create_codewords(datapath):
    # convert to codewords
    f = open(datapath, 'r')
    text_array = list(f.read())
    
    codeword_array = []
    n = len(text_array)
    for i in range(n):
        # convert text to ascii int 
        ascii_int = ord(text_array[i])
        # convert int to 8 bit
        ascii_bits = '{0:08b}'.format(ascii_int)
        
        codeword_array.append(str(ascii_bits))
    
    save_file("codewords", codeword_array)
    
    
def create_phi():
    
    # set parameters
    beta = 1/2
    f_sample = 22050
    T = 1/f_sample
    num_s = int(sampling_rate)                        # change if needed 
    num_s_h = int(num_s/2)                  # need to be not multiple of 2 otherwise devision by zero!
      
        
    #create phi (root raised cosine)
    lengths_phi = num_s
    phi = [0]*(lengths_phi)
    for i in range(-int(lengths_phi/2), int(lengths_phi/2)):
        t = T/num_s_h*i
            
        # implementation of root raised cosine function
        term_plus = (1+beta)*(t)/T
        term_minus = (1-beta)*(t)/T
        sinc_term = np.sinc(term_minus)
        denomitor = 1-math.pow((4*beta*(t)/T),2)


        phi[i+int(lengths_phi/2)] = 4*beta/(math.pi*math.sqrt(T))*(math.cos(term_plus*math.pi)+(1-beta)*math.pi/(4*beta)*sinc_term)/(denomitor)

        phi[i+int(lengths_phi/2)] = phi[i+int(lengths_phi/2)]/float(168)
        
    save_file("phi_testing", phi)
        

# waveform former creates signal based on codewords
# codewords are send as one large array 

def waveform_former():
    
    start = time.clock()
    # create codewords
    codewords = open_file("codewords")
    #codeword = "1111111111111111"            # Syncronization Pattern at beginning!
    codeword = ""
    for i in range(len(codewords)):
        codeword = codeword + str(codewords[i])
        
    end_indicator = "22111111111111111111111111"
    codeword = codeword + end_indicator      # Syncronization Pattern at end of string!
    
    codeword = codeword.replace(" ", "")
    print("Creating waveform...")
    print("")
    # set parameters
    beta = 1/2
    f_sample = 22050
    T = 1/f_sample
    num_s = int(sampling_rate)                        # change if needed 
    num_s_h = int(num_s/2)                  # need to be not multiple of 2 otherwise devision by zero!
      
    
    num_bits = len(codeword)                # number of bits transmitted
    lengths_w = (num_bits+1)*num_s
    w = [0]*(lengths_w)                     # create w
     
    #create_phi()
    phi = open_file("phi_testing")
    phi = list(map(float, phi))

    
    c = np.asarray(list("".join(codeword))).astype(np.float)
    c = np.where(c==0, -1, c)
    c = np.where(c==2, 0, c)
    w = np.kron(c, np.asarray(phi))
    
    
    """
    # compute waveform
    for i in range(num_bits):
        c = codeword[i]
        w_temp = [0]*(lengths_w)
        
        if c == '0':
            var_c = -1
            var_codeword = num_s*i
            w_temp[var_codeword:(len(phi)+var_codeword)] = list(map(operator.sub, w_temp,phi))

        else:
            var_c = 1
            var_codeword = num_s*i
            w_temp[var_codeword:(len(phi)+var_codeword)] = list(map(operator.add, w_temp,phi))
            
        w = list(map(operator.add, w,w_temp))
    """
    
    
    starting_seq = [1]*sampling_rate*8
    #ending_buffer = [0]*302
    
    w = starting_seq + w.tolist() 
    
    #w = starting_seq + w + ending_seq
    
    #save_file("test", w)
    
    # convert signal to txt file for output
    #save_file("waveform", w)
    
    
    #print (w[:100])
    end = time.clock()
    
    print("Time for function waveform_former:")
    print(end-start)
    print("")
    
    passband_filter(w)

def passband_filter(w):
    
    # create signal 
    
    #w = open_file("waveform")
   
    lengths_w = len(w)

    f_c_1 = float(2000)
    f_c_2 = float(4000)

    x1 = [0]*lengths_w
    x2 = [0]*lengths_w
    a = int(lengths_w/2)

    for n in range(-a,a):
        t = (1/22050)*n        # check if correct!
        x1[n+a] = math.sqrt(2)*float(w[n+a])*math.cos(2*math.pi*f_c_1*t)
        x2[n+a] = math.sqrt(2)*float(w[n+a])*math.cos(2*math.pi*f_c_2*t)
    x = np.concatenate((x1,x2))
    start = time.clock()
    save_file("passband", x)
    end = time.clock()
    print("Time for function passband:")
    print(end-start)
    print("")
    
    
    """
    fourier_passband = np.square(fft(x))
    th = (abs(fourier_passband) > 0.6*10**11)

    first_max_value = np.where(th == True)[0][0]
    print("Frequency channel: ")
    frequency = first_max_value/lengths_w*22050
    print(frequency)
    print("")
    """
    
        
    
def save_file(name, data):
    data_name = str(name) + ".txt"
    np.savetxt(data_name, data, delimiter="\n", fmt="%s")

    
def open_file(name):
    data_name = str(name) + ".txt"
    f = open(data_name, 'r')
    data = f.read().split('\n')
    data.pop()
    
    return data
    
    

    