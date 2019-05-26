# reciever


import numpy as np
import math
import copy
from scipy import fft
from transmitter import save_file, open_file
import binascii
import time
import operator


def compute_sync_w():
    phi = open_file("phi_testing")
    phi = list(map(float, phi))
    sync_w = np.zeros(8*len(phi))
    
    for i in range(8):
        sync_w[(i*len(phi)):((i+1)*len(phi))] = phi

    return sync_w

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
    
    
    
def lowpass_filter():
   

    start = time.clock()

    w = open_file("output")
    w = np.array(w).astype(np.float)

    
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
    print("")
    
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
    
    
    
    sync_w = compute_sync_w()
    
    output = np.convolve(R,sinc_func)    
    
    
    codewords = open_file("codewords")
    num_bits = len(codewords)*8
    
    sync_test = np.convolve(output, sync_w)
    
    mirrow = False
    max_index = np.argmax(sync_test)
    min_index = np.argmin(sync_test)
    if abs(sync_test[min_index]) > abs(sync_test[max_index]):
        max_index = min_index
        mirrow = True
    num_s = int(302)
    start_index = max_index 
     
    output = output[start_index:(start_index+num_s*num_bits)]
    save_file("test", output) 
    
    inner_product(output, num_bits, mirrow)
    
def inner_product(r, num_bits, mirrow):
    
    start = time.clock()
    
    
    # devide r into chunks
    r_chunks = []
    n = int(302)
    for i in range(0, len(r), n):
        r_chunks.append(r[i:(i + n)])
    
   
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
       
        if y_temp_plus > y_temp_minus:      # changed!!!
            if mirrow:
                y.append(0)
            else:
                y.append(1)
        else:
            if mirrow:
                y.append(1)
            else:
                y.append(0)
   
    save_file("y", y)
    
    end = time.clock()
    print("Time for function inner_product:")
    print(end-start)
    print("")
    
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
    print("")
    
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
    print("")
    
    cleartext = open("text.txt", "r")
    print("Original text: ")
    print(cleartext.read())
    print("")

  
    
def run():
    lowpass_filter()
    check()

    