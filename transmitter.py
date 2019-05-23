# pdc project


import numpy as np
import math
import operator
from scipy.fftpack import fft, ifft, fftshift
import time


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
        
    #print(codeword_array)
    save_file("codewords", codeword_array)
   

        

# waveform former creates signal based on codewords
# codewords are send as one large array 

def waveform_former():
    
    start = time.clock()
    # create codewords
    codewords = open_file("codewords")
    codeword = ""
    for i in range(len(codewords)):
        codeword = codeword + str(codewords[i])
    codeword = codeword.replace(" ", "")
    #print(codeword)
    
    # set parameters
    beta = 1/2
    f_sample = 22050
    T = 1/f_sample
    num_s = int(502)                        # change if needed 
    num_s_h = int(num_s/2)                  # need to be not multiple of 2 otherwise devision by zero!
      
    
    num_bits = len(codeword)                # number of bits transmitted
    lengths_w = (num_bits+1)*num_s
    w = [0]*(lengths_w)                     # create w
     
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

        
    save_file("phi_testing", phi)
    
    # compute waveform
    for i in range(num_bits):
        c = codeword[i]
        w_temp = [0]*(lengths_w)
         
        if float(codeword[i]) == 0:
            var_c = -1
            var_codeword = num_s*i
            w_temp[var_codeword:(len(phi)+var_codeword)] = list(map(operator.sub, w_temp,phi))
        else:
            var_c = 1
            var_codeword = num_s*i
            w_temp[var_codeword:(len(phi)+var_codeword)] = list(map(operator.add, w_temp,phi))
            
        w = list(map(operator.add, w,w_temp))
    
    # convert signal to txt file for output
    save_file("waveform", w)
    
    end = time.clock()
    
    print("Time for function waveform_former:")
    print(end-start)
    
    passband_filter(lengths_w)

def passband_filter(lengths_w):
    
    # create signal 
    start = time.clock()
    w = open_file("waveform")
   
    f_c = float(2000)

    x = [0]*lengths_w
    a = int(lengths_w/2)
    for n in range(-a,a):
        t = (1/22050)/10*n        # check if correct!
        x[n+a] = math.sqrt(2)*float(w[n+a])*math.cos(2*math.pi*f_c*t)
       
    
    save_file("passband", x)
    
    fourier_passband = np.square(fft(x))
    max_value_index = np.argmax(fourier_passband)
    print("Frequency channel: ")
    print(abs(max_value_index - len(fourier_passband)/2))
    
    save_file("passband_fourier", fourier_passband)
    
    end = time.clock()
    print("Time for function passband:")
    print(end-start)
    
# DELETE AFTER FINISH
    
def test():

    # ------------- Compute Phi(t) -------------
    # Use as base function root raised cosine
    T = 1/22050 # maybe *5
    num_s_h = 5
    beta = 1/2
    
    phi = [0]*(2000)
    for j in range(-1000,1000):
        t = T/num_s_h*j

        # implementation of root raised cosine function
        term_plus = (1+beta)*(t)/T
        term_minus = (1-beta)*(t)/T
        sinc_term = np.sinc(term_minus)
        denomitor = 1-math.pow((4*beta*(t)/T),2)

        phi[j+1000] = 4*beta/(math.pi*math.sqrt(T))*(math.cos(term_plus*math.pi)+(1-beta)*math.pi/(4*beta)*sinc_term)/(denomitor)

    
    save_file("phi_before", phi)
    
    
    
    # ------------- Fourier Phi(t) -------------
    
    fourier_phi = np.square(fft(phi))
    #fourier_x = np.square(fftshift(fourier_x))
    #fourier_x = np.square(fft(x)) 
    
    save_file("passband_before", fourier_phi)

    
    # ------------- Shift Phi(t) -------------
    
    phi = open_file("phi_before")
    
    f_c = float(2000)
    x = [0]*len(phi)
  
    for n in range(-1000,1000):
        t = (1/22050)*n # has to be the same as the sampling steps! #/5
        #t = n
        x[n+1000] = float(phi[n+1000])*math.sqrt(2)*math.cos(2*math.pi*f_c*t)
        
    
    save_file("phi_after", x)
 
    
    # ------------- Compute Fourier of shifted Phi(t) -------------
    
    fourier_x_t = np.square(fft(x))
     
    save_file("passband_after", fourier_x_t)
    
    
    
def save_file(name, data):
    data_name = str(name) + ".txt"
    fs = ""
    for i in range(len(data)):
        fs = fs + str(np.real(data[i])) + "\n"
    ff = open(data_name, "w")
    ff.write(fs)
    ff.close() 
    
def open_file(name):
    data_name = str(name) + ".txt"
    f = open(data_name, 'r')
    data = f.read().split('\n')
    data.pop()
    
    return data
    
    

    