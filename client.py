# ############################################################################
# client.py
# =========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Black-box channel simulator. (client)

Instructions
------------
python3 client.py --input_file=[FILENAME] --output_file=[FILENAME] --srv_hostname=[HOSTNAME] --srv_port=[PORT]

python client.py --input_file=final_signal.txt --output_file=output.txt --srv_hostname=iscsrv72.epfl.ch --srv_port=80
"""


import argparse
import pathlib
import socket

import numpy as np

import channel_helper as ch

def parse_args():
    parser = argparse.ArgumentParser(description="COM-303 black-box channel simulator. (client)",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--input_file', type=str, required=True,
                        help='.txt file containing (N_sample,) rows of float samples.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='.txt file to which channel output is saved.')
    parser.add_argument('--srv_hostname', type=str, required=True,
                        help='Server IP address.')
    parser.add_argument('--srv_port', type=int, required=True,
                        help='Server port.')

    args = parser.parse_args()

    args.input_file = pathlib.Path(args.input_file).resolve(strict=True)
    if not (args.input_file.is_file() and
            (args.input_file.suffix == '.txt')):
        raise ValueError('Parameter[input_file] is not a .txt file.')

    args.output_file = pathlib.Path(args.output_file).resolve(strict=False)
    if not (args.output_file.suffix == '.txt'):
        raise ValueError('Parameter[output_file] is not a .txt file.')

    return args

if __name__ == '__main__':
    args = parse_args()
    tx_p_signal = np.loadtxt(args.input_file)

    N_sample = tx_p_signal.size
    if not ((tx_p_signal.shape == (N_sample,)) and
            np.issubdtype(tx_p_signal.dtype, np.floating)):
        raise ValueError('Parameter[input_file] must contain a real-valued sequence.')

    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as sock_cl:
        sock_cl.connect((args.srv_hostname, args.srv_port))
        ch.send_ndarray(sock_cl, tx_p_signal)

        rx_p_signal = ch.recv_ndarray(sock_cl)
        np.savetxt(args.output_file, rx_p_signal)
