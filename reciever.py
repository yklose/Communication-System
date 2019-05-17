# reciever


import numpy as np
import math
import copy



def filter_noise():
    
    datapath = "output.txt"
    f = open(datapath, 'r')
    signal = f.read().split('\n')
    signal.pop()
    
    copy_signal = np.array(signal).astype(np.float)

    th = copy_signal > 0.5
    start_index = np.where(th == True)[0][0]
    end_index = np.where(th == True)[0][-1]
    
    copy_signal[0:start_index] = 0
    copy_signal[end_index:] = 0
    
    
    first_filter_stage = ""
    for i in range(len(copy_signal)):
        first_filter_stage = first_filter_stage + str(copy_signal[i]) + "\n"
    first_filter_stage_file = open("filtered_noise.txt", "w")
    first_filter_stage_file.write(first_filter_stage)
    first_filter_stage_file.close()
    
def lowpass_filter():
   
    
    f = open("filtered_noise.txt", 'r')
    w = f.read().split('\n')
    w.pop()
    
    f_c = float(2000)
    R = [0]*len(w)
    for n in range(len(w)):
        # check if correct!
        R[n] = math.sqrt(2)*float(w[n])*math.cos(2*math.pi*f_c*n/22050)
    
    
    # centered sinc
    sinc_func = [0]*len(w)
    for n in range(len(sinc_func)):
        w_sinc = n-len(w)/2
        if w_sinc == 0:
            sinc_func[n] = 1
        else:
            sinc_func[n] = math.sin(math.pi*w_sinc*f_c/22050)/(math.pi*w_sinc*f_c/22050)
    
    #print(sinc_func)
    
    output = np.convolve(R,sinc_func)    
    
    
    final_signal = ""
    for i in range(len(output)):
        final_signal = final_signal + str(output[i]) + "\n"
    signal_file = open("lowpass.txt", "w")
    signal_file.write(final_signal)
    signal_file.close()

    
def run():
    filter_noise()
    lowpass_filter()
    