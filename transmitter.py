# pdc project


import numpy as np

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
        
    
    print(codeword_str)
    
    return codeword_str
    # save the file
    
    
# waveform former!

def waveform_former(codewords):
    
    codewords = create_codewords(datapath)
    
    # root raised cosine? 
    # per bit or whole message in once?
    
    # within certain frequency in order to go through the filter
    signal = ...
    
    signal_file = open("signal.txt", "w")
    signal_file.write(signal)
    signal_file.close()


#to do: plot the recieved signal!
    

    