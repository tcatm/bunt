# python2

import gtk
import socket
import math
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

csd = gtk.ColorSelectionDialog('Gnome Color Chooser')
cs = csd.colorsel
cs.set_has_opacity_control(False)

def color_changed(foo):
    c = cs.get_current_color()
    data = cal.get_rgb(c.red/256.0, c.green/256.0, c.blue/256.0)
    data = [data[1], data[0], data[2]]
    setleds(data * int(sys.argv[2]))

cs.connect("color-changed", color_changed)

if csd.run()!=gtk.RESPONSE_OK:
   print('No color selected.')
   sys.exit()

