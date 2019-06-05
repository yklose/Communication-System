# ############################################################################
# client.py
# =========
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

import struct
import numpy as np
import io

def send_ndarray(sock, data):
    """
    Send a NumPy array over the network.

    Parameters
    ----------
    sock : :py:class:`~socket.socket`
    data : :py:class:`~numpy.ndarray`
    """
    with io.BytesIO() as f:
        np.save(f, data)
        byte_data = f.getvalue()

    # Pack message length
    msg = struct.pack('>I', len(byte_data)) + byte_data
    sock.sendall(msg)

def recv_ndarray(sock):
    """
    Receive a NumPy array from the network.

    Parameters
    ----------
    sock : :py:class:`~socket.socket`

    Returns
    -------
    data : :py:class:`~numpy.ndarray`
    """
    # Extract message length
    N_msg_raw = recv_bytes(sock, 4)
    N_msg = struct.unpack('>I', N_msg_raw)[0]

    msg = recv_bytes(sock, N_msg)
    with io.BytesIO(msg) as f:
        data = np.load(f)
    return data

def recv_bytes(sock, N_byte):
    """
    Receive bytes from the network.

    Parameters
    ----------
    sock : :py:class:`~socket.socket`
    N_byte : int
        Number of bytes to read.

    Returns
    -------
    byte_data : bytes
    """
    packet_size = 2 ** 12

    packets, N_byte_read = [], 0
    while N_byte_read < N_byte:
        packet = sock.recv(min(packet_size, N_byte - N_byte_read))
        packets.append(packet)
        N_byte_read += len(packet)

    byte_data = b''.join(packets)
    return byte_data
