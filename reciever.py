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
    
    #print(end_index - start_index)
    
    copy_signal[0:start_index] = 0
    copy_signal[end_index:] = 0
    
    save_file("filtered_noise", copy_signal)
    
    
def lowpass_filter():
   
    start = time.clock()
    w = open_file("passband") # usually filtered_noise.txt
    
    # back to zero frequency band
    f_c = float(2000)
    R = np.zeros(len(w))
    a = int(len(w)/2)
    # centered sinc
    sinc_func = np.zeros(len(w))
    for n in range(-a,a):
        t = (1/22050)/10*n  
        
        R[n+a] = math.sqrt(2)*float(w[n+a])*math.cos(2*math.pi*f_c*t)
        
        if n == 0:
            sinc_func[n+a] = 1
        else:
            sinc_func[n+a] = math.sin(math.pi*t*f_c)/(math.pi*t*f_c)
   
        
      
    save_file("sinc", sinc_func)
    
    output = np.convolve(R,sinc_func)    

    save_file("lowpass", output)
    
    
    # ------------- Compute Fourier of sinc(t) -------------
    
    sinc_fourier = np.square(fft(sinc_func))
    
    save_file("fourier_sinc", sinc_fourier)

    end = time.clock()
    print("Time for function lowpass_filter:")
    print(end-start)
    
def inner_product():
    
    start = time.clock()
    codewords = open_file("codewords")
    num_bits = len(codewords)*8
    
    
    # open R(t)
    r = open_file("lowpass") #actually lowpass.txt for test purpose waveform.txt
    copy_r = np.array(r).astype(np.float)

    th = (abs(copy_r) > 7500/2)                        # check threshold for diffent texts
    start_index = np.where(th == True)[0][0] - 140
    end_index = np.where(th == True)[0][-1] + 140
    
    r = r[start_index:end_index]
    save_file("cut_lowpass", r)
    
    # devide r into chunks
    r_chunks = []
    n = 502
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
    #filter_noise()
    lowpass_filter()
    inner_product()
    check()

    