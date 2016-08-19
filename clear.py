import socket
import math
import colorsys
from array import array
from time import sleep

UDP_IP = "2a01:170:1112:0:bad8:12ff:fe66:fa1"
UDP_PORT = 2812

sock = socket.socket(socket.AF_INET6,socket.SOCK_DGRAM)

def setleds(data):
    data = [int(x) % 256 for x in data]
    sock.sendto(array('B', data).tostring(), (UDP_IP, UDP_PORT))

def wheel(pos):
    pos = 255 - pos
    pos = pos % 256
    if pos < 85:
        return [255 - pos * 3, 0, pos * 3]
    elif pos < 170:
        pos -= 85
        return [0, pos * 3, 255 - pos * 3]
    else:
        pos -= 170
        return [pos * 3, 255 - pos * 3, 0]

def rgb2grb(data):
    return (data[1], data[0], data[2])

import sys

setleds([sys.argv[2], sys.argv[1], sys.argv[3]] * 219 + 200 * [0,0,0])
