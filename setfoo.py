# python2

import socket
import math
import colorsys
import sys
from array import array
from time import sleep

from cal import LEDCalibration

UDP_IP = "2a01:170:1112:0:bad8:12ff:fe66:fa1"
UDP_PORT = 2812

sock = socket.socket(socket.AF_INET6,socket.SOCK_DGRAM)

def setleds(data):
    data = [int(x) % 256 for x in data]
    sock.sendto(array('B', data).tostring(), (UDP_IP, UDP_PORT))


cal = LEDCalibration(sys.argv[1])

(r, g, b) = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
data = cal.get_rgb(r, g, b)
data = [data[1], data[0], data[2], 0,0,0,0,0,0,0,0,0]
setleds(data * 119)
