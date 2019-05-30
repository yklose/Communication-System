# reciever


import numpy as np
import math
import copy
from scipy import fft
import binascii
import time
import operator


def compute_sync_w(select, frequency_shift):
    if select == 1:
        f_c = 2000
    else:
        f_c = 4000
        
    phi = open_file("phi_testing")
    phi = list(map(float, phi))

    sync_w = np.zeros(24*len(phi))

    
    for i in range(24):
     
        # the known syncronization starting pattern is 1111111111111111
        sync_w[(i*len(phi)):((i+1)*len(phi))] = phi

    # shift in frequency
    if frequency_shift:
        a = int(len(sync_w)/2)
        for n in range(-a,a):
            t = (1/22050)*n  
            sync_w[n+a] = math.sqrt(2)*float(sync_w[n+a])*math.cos(2*math.pi*f_c*t)
    
    return sync_w

def compute_sinc(w):
 
    f_c_1 = float(2000)
    f_c_2 = float(4000)
    a = int(len(w)/20) 
    sinc_func1 = np.zeros(int(len(w)/10))
    sinc_func2 = np.zeros(int(len(w)/10))
    
    for n in range(-a,a):
        t = (1/22050)*n  
        if n == 0:
            sinc_func1[n+a] = 1
            sinc_func2[n+a] = 1
        else:
            sinc_func1[n+a] = math.sin(math.pi*t*f_c_1)/(math.pi*t*f_c_1)
            sinc_func2[n+a] = math.sin(math.pi*t*f_c_2)/(math.pi*t*f_c_2)
                
    save_file("sinc1", sinc_func1)
    save_file("sinc2", sinc_func2)
    
    
    
def lowpass_filter():
   
    print("Reconstructing Signal...")
    print("")

    w = open_file("output")
    w = np.array(w).astype(np.float)

    # centered sinc (call when carrier frequency changes!)
    #compute_sinc(w)
   
    
    sinc_func1 = open_file("sinc1")
    sinc_func1 = list(map(float, sinc_func1))
    
    sinc_func2 = open_file("sinc2")
    sinc_func2 = list(map(float, sinc_func2))
 
    
    R = np.zeros(len(w))
    
    start = time.clock()
    
    sync_w1 = compute_sync_w(1, True)
    sync_w2 = compute_sync_w(2, True)
    
    sync_test1 = np.absolute(np.convolve(w, sync_w1))
    sync_test2 = np.absolute(np.convolve(w, sync_w2))
    
    #save_file("test1", sync_test1) 
    #save_file("test2", sync_test2) 
    
    max_index1 = np.argmax(sync_test1)
    max_index2 = np.argmax(sync_test2)
    
    if sync_test1[max_index1] < sync_test2[max_index2]:
        print("In 4000 Hz Frequency")
        print("")
        sync_test = sync_test2
        sinc_func = sinc_func2
        select = 2
        f_c = float(4000)
    else:
        print("In 2000 Hz Frequency")
        print("")
        sync_test = sync_test1
        sinc_func = sinc_func1
        select = 1
        f_c = float(2000)
    
    end = time.clock()
    print("Time for frequency Selection:")
    print(end-start)  
    print("")
    
    start = time.clock()
    
    # shift back in frequency
    a = int(len(w)/2)
    for n in range(-a,a):
        t = (1/22050)*n  
        
        R[n+a] = math.sqrt(2)*float(w[n+a])*math.cos(2*math.pi*f_c*t)
    
    # finding start index
    #sync_w = compute_sync_w(select, False, False) 
    sync_w = np.ones(302*8)
    output = np.convolve(R,sinc_func)  
    sync_test = np.convolve(output, sync_w)
    
    save_file("test1", sync_test) 
    #save_file("test", sync_test) 
    mirrow = False
    max_index = np.argmax(sync_test)
    min_index = np.argmin(sync_test)
    if abs(sync_test[min_index]) > abs(sync_test[max_index]):
        max_index = min_index
        mirrow = True
        #print("Mirrowed")
    num_s = int(302)
    start_index = max_index 
    
    # finding end index
    #sync_w_end = compute_sync_w(select, False, True)
    copy_output = output[start_index:]
    sync_w_end = compute_sync_w(select, False)
    sync_test_end = np.convolve(copy_output, sync_w_end)
    
    save_file("test2", sync_test_end) 
    
    max_index_end = np.argmax(sync_test_end) + start_index
    min_index_end = np.argmin(sync_test_end) + start_index
    if mirrow:
        max_index_end = min_index_end
    num_s = int(302)
    end_index = max_index_end
    
    print(end_index)
    
    lengths_message = int(round((end_index-start_index)/302 - 26))
      
    output = output[start_index:(start_index+lengths_message*num_s)]
    

    end = time.clock()
    print("Time for finding startpoint and endpoint in time domain:")
    print(end-start)  
    print("")
    
    
    inner_product(output, lengths_message, mirrow)
    
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

    print("Done...")
  
    
def run():
    lowpass_filter()
    check()

    
    
def save_file(name, data):
    data_name = str(name) + ".txt"
    np.savetxt(data_name, data, delimiter="\n", fmt="%s")

    
def open_file(name):
    data_name = str(name) + ".txt"
    f = open(data_name, 'r')
    data = f.read().split('\n')
    data.pop()
    
    return data
    