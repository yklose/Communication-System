# reciever


import numpy as np
import math
import copy
from scipy import fft
from transmitter import save_file, open_file
import binascii
import time

def filter_noise():
    
    signal = open_file("output")
    
    copy_signal = np.array(signal).astype(np.float)

    th = (abs(copy_signal) > 0.5)
    start_index = np.where(th == True)[0][0] - 1
    end_index = np.where(th == True)[0][-1] + 1

    
    copy_signal[0:start_index] = 0
    copy_signal[end_index:] = 0
   
    
    lowpass_filter(copy_signal)
    
def compute_sinc(w):
 
    f_c = float(4000)
    a = int(len(w)/2)
    sinc_func = np.zeros(len(w))
    
    for n in range(-a,a):
        t = (1/22050)*n  
        if n == 0:
            sinc_func[n+a] = 1
        else:
            sinc_func[n+a] = math.sin(math.pi*t*f_c)/(math.pi*t*f_c)
                
    save_file("sinc", sinc_func)
    
    
    
def lowpass_filter(w):
   
    start = time.clock()

    sinc_func = open_file("sinc")
    sinc_func = list(map(float, sinc_func))
 
    f_c = float(4000)
    R = np.zeros(len(w))
    a = int(len(w)/2)
    
    # centered sinc (call when carrier frequency changes!)
    #compute_sinc(w)
    
    end = time.clock()
    print("Time beginning:")
    print(end-start)  
    
    """
    # for vectorizing
    n = np.arange(-a,a+1)
    t = (1/22050)/10*n 
    n_plus_a = a+500
    
    R = math.sqrt(2)*np.take(w, n_plus_a).astype(np.float)*np.cos(2*math.pi*f_c*t)
    """
    
    for n in range(-a,a):
        t = (1/22050)*n  
        
        R[n+a] = math.sqrt(2)*float(w[n+a])*math.cos(2*math.pi*f_c*t)
       
    
    output = np.convolve(R,sinc_func)    

    inner_product(output)
    
def inner_product(r):
    
    start = time.clock()
    codewords = open_file("codewords")
    num_bits = len(codewords)*8
   
    copy_r = np.array(r).astype(np.float)

    th = (abs(copy_r) > 2)                        # check threshold for diffent texts
    start_index = np.where(th == True)[0][0] - 140
    end_index = np.where(th == True)[0][-1] + 140
    
    r = r[start_index:end_index]
    #save_file("cut_lowpass", r)
    
    # devide r into chunks
    r_chunks = []
    n = 302
    for i in range(0, len(r), n):
        r_chunks.append(r[i:i + n])
    r_chunks.pop()
   
    # open phi(t)
    phi = open_file("phi_testing")
    
    
    # matched filter
    y = []
    for i in range(num_bits):
        r_array = np.asarray(r_chunks[i])
        r_array = list(map(float, r_array))
        phi_plus = list(map(float, phi))
        phi_minus = np.negative(phi_plus)
        
      
        y_temp_plus = np.dot(r_array, phi_plus)
        y_temp_minus = np.dot(r_array, phi_minus)
       
        if y_temp_plus > y_temp_minus:
            y.append(1)
        else:
            y.append(0)
   
    save_file("y", y)
    
    end = time.clock()
    print("Time for function inner_product:")
    print(end-start)
    
def check():
    # check results
    y = open_file("y")
    y = ''.join(y)
    codewords = open_file("codewords")
    codewords = ''.join(codewords)
    
    counter = 0
    for i in range(len(y)):
        if y[i] == codewords[i]:
            counter+=1
    print("Number of Correct Bits: " + str(counter) + "/" + str(len(y)))
    
    # Encode Bits
    y_bytes = []
    n = 8
    for i in range(0, len(y), n):
        bit_string = y[i:i + n]
        decimal = int(bit_string, 2)
        y_bytes.append(str(chr(decimal)))
    y_bytes = ''.join(y_bytes)  
    print("Decoded recieved signal: ")
    print(y_bytes)
    
    cleartext = open("text.txt", "r")
    print("Original text: ")
    print(cleartext.read())

  
    
def run():
    filter_noise()
    check()

    