# pdc project


import numpy as np
import math
import operator
from scipy.fftpack import fft, ifft, fftshift



def run():
    datapath = "text.txt"
    waveform_former(datapath)
    test()

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
    print("Codeword: ")
    print(codeword_str)
    return codeword_str
        

# waveform former creates signal based on codewords
# codewords are send as one large array (concatenade)

def waveform_former(datapath):
    
    # create codewords
    codewords = create_codewords(datapath)
    codeword = codewords.replace(" ", "")
    
    # set parameters
    beta = 1/2
    f_sample = 22050
    T = 1/f_sample
    num_s = int(10)                         # change if needed 
    num_s_h = int(num_s/2)                  # need to be not multiple of 2 otherwise devision by zero!
      
    
    #codeword = [1,1,1,0,0]                 # just for testing! 
    num_bits = len(codeword)                # number of bits transmitted
    lengths_w = (num_bits+1)*num_s
    w = [0]*(lengths_w)                     # create w
    
    # use root raised cosine functions!
    # maybe improvement: compute phi once on large interval, paste!
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
        
                
            phi = 4*beta/(math.pi*math.sqrt(T))*(math.cos(term_plus*math.pi)+(1-beta)*math.pi/(4*beta)*sinc_term)/(denomitor)
            
            if float(codeword[i]) == 0:
                var_c = -1
            else:
                var_c = 1
            
            w_temp[(j+num_s_h+num_s*i)] = float(var_c)*phi
            
        
        # add the w_temp to the signal (w = w + w_temp)
        # print("Step: " + str(i+1) + "/" + str(num_bits))
        w = list(map(operator.add, w,w_temp))
    
   
    # convert signal to txt file for output
    signal = ""
    for i in range(len(w)):
        signal = signal + str(w[i]) + "\n"
    signal_file = open("waveform.txt", "w")
    signal_file.write(signal)
    signal_file.close()
    
    passband_filter(lengths_w)

def passband_filter(lengths_w):
    
    # create signal (saved as signal.txt)
    #waveform_former()
    
    f = open("waveform.txt", 'r')
    w = f.read().split('\n')
    w.pop()
    
    f_c = float(2000)
    x = [0]*lengths_w
    for n in range(lengths_w):
        x[n] = math.sqrt(2)*float(w[n])*math.cos(2*math.pi*f_c*n/22050)
        
    final_signal = ""
    for i in range(len(x)):
        final_signal = final_signal + str(x[i]) + "\n"
    signal_file = open("passband.txt", "w")
    signal_file.write(final_signal)
    signal_file.close()  

    
def test():

    # ------------- Compute Phi(t) -------------
    # Use as base function root raised cosine
    T = 1/22050*5
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

         
    phi_signal_f = ""
    for i in range(len(phi)):
        phi_signal_f = phi_signal_f + str(np.real(phi[i])) + "\n"
    phi_file_f = open("phi.txt", "w")
    phi_file_f.write(phi_signal_f)
    phi_file_f.close()
    
    
    # ------------- Fourier Phi(t) -------------
    
    fourier_x = fft(phi)
    fourier_x = np.square(fftshift(fourier_x))
     
    #fourier_x = np.square(fft(x)) 
    final_signal = ""
    for i in range(len(fourier_x)):
        final_signal = final_signal + str(np.real(fourier_x[i])) + "\n"
    signal_file = open("passband_test.txt", "w")
    signal_file.write(final_signal)
    signal_file.close() 
    
    # ------------- Shift Phi(t) -------------
    
    f = open("phi.txt", 'r')
    w = f.read().split('\n')
    w.pop()
    
    f_c = float(2000)
    x = [0]*len(phi)
    for n in range(len(phi)):
        x[n] = math.sqrt(2)*float(w[n])*math.cos(2*math.pi*f_c*n/22050)
        
    final_signal = ""
    for i in range(len(x)):
        final_signal = final_signal + str(x[i]) + "\n"
    signal_file = open("passband.txt", "w")
    signal_file.write(final_signal)
    signal_file.close() 
    
    # ------------- Compute Fourier of shifted Phi(t) -------------
    
    fourier_x_t = fft(x)
    fourier_x_t = np.square(fftshift(fourier_x_t))
     
    fs = ""
    for i in range(len(fourier_x)):
        fs = fs + str(np.real(fourier_x_t[i])) + "\n"
    ff = open("passband_test_1.txt", "w")
    ff.write(fs)
    ff.close() 
    
    
    
    
    

    