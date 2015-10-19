import numpy as np
import socket
import math
import colorsys
from array import array
from time import sleep, clock
from pylut import *

UDP_IP = "2a01:170:1112:0:bad8:12ff:fe66:fa1"
UDP_PORT = 2812

sock = socket.socket(socket.AF_INET6,socket.SOCK_DGRAM)

def setleds(lut, data):
    """
    data = [(r, g, b), (r, g, b), ...]
    """
    data = [lut.ColorFromColor(Color.FromRGBInteger(x[0], x[1], x[2], 8)).ToRGBIntegerArray(8) for x in data]

    data = [(x[1], x[0], x[2]) for x in data]
    data = [item for sublist in data for item in sublist]
    sock.sendto(array('B', data).tostring(), (UDP_IP, UDP_PORT))

import pyaudio
import struct
import wave

FPS = 30

nFFT = 2048
BUF_SIZE = 1 * nFFT
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 96000

FALL = 0.71
RISE = 0.95
OVERSHOOT = 0.3

state = dict(l=np.zeros(119), r=np.zeros(119))

fact_cache = {}
def fact(n):
  ''' Memoized factorial function '''
  try:
    return fact_cache[n]
  except(KeyError):
    if n == 1 or n == 0:
      result = 1
    else:
      result = n*fact(n-1)
    fact_cache[n] = result
    return result


def bernstein(t,n,i):
  ''' Bernstein coefficient '''
  binom = fact(n)/float(fact(i)*fact(n - i))
  return binom*((1-t)**(n-i))*(t**i)


def bezier_gradient(colors):
  ''' Returns a "bezier gradient" dictionary
      using a given list of colors as control
      points. Dictionary also contains control
      colors/points. '''
  # RGB vectors for each color, use as control points
  RGB_list = colors
  n = len(RGB_list) - 1

  def bezier_interp(t):
    ''' Define an interpolation function
        for this specific curve'''
    # List of all summands
    summands = [
      map(lambda x: int(bernstein(t,n,i)*x), c)
      for i, c in enumerate(RGB_list)
    ]
    # Output color
    out = [0,0,0]
    # Add components of each summand together
    for vector in summands:
      for c in range(3):
        out[c] += vector[c]

    return out

  return bezier_interp

grad = bezier_gradient([[0,0,0], [10,0,10], [255,0,0], [255,150,150], [255,100,0], [255, 255, 0], [255, 130, 0], [255, 255, 200]])
#grad = bezier_gradient([[0,0,0], [0,0,0], [80,0,80], [80,0,80], [0,255,0], [0,40,0], [0, 50, 0], [0, 80, 0]] * 2)
#grad = bezier_gradient([[0,0,0], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 255, 255]])
#grad = bezier_gradient([[0, 0, 0], [255, 255, 255]])

def animate(lut, stream, MAX_y, state):
  # Read n*nFFT frames from stream, n > 0
  N = max(stream.get_read_available() / nFFT, 1) * nFFT
  data = stream.read(N)

  # Unpack data, LRLRLR...
  y = np.array(struct.unpack("%dh" % (N * CHANNELS), data)) / MAX_y
  y_L = y[::2]
  y_R = y[1::2]

  Y_L = np.fft.fft(y_L, nFFT)
  Y_R = np.fft.fft(y_R, nFFT)

#  print(list(np.fft.fftfreq(nFFT, 1.0/RATE)[0:nFFT/2]))

  Y_L = Y_L[0:nFFT/2]
  Y_R = Y_R[0:nFFT/2]

  N = 0.1

  Y_L = N * abs(Y_L)
  Y_R = N * abs(Y_R)

  l_l = np.zeros(119)
  l_r = np.zeros(119)

  for i in range(0, 119):
    x = int((i*i)/(math.pow(119,2))* nFFT/16)
    y = int(((i+1)*(i+1))/(math.pow(119,2))* nFFT/16) + 1
    l_l[i] = sum(Y_L[x:y])
    l_r[i] = sum(Y_R[x:y])

  l_l = np.minimum(1, l_l)
  l_l = (1 - RISE) * state['l'] + RISE * np.maximum(FALL * state['l'], l_l) + OVERSHOOT * l_l
  state['l'] = l_l

  l_r = np.minimum(1, l_r)
  l_r = (1 - RISE) * state['r'] + RISE * np.maximum(FALL * state['r'], l_r) + OVERSHOOT * l_r
  state['r'] = l_r

  data = [grad(min(1, x)) for x in 1 * np.hstack((np.flipud(l_r), l_l))]

  setleds(lut, data)


def main():
  lut = LUT.FromCubeFile("ws2812b-from-srgb.cube")

  setleds(lut, [[0,0,0]] * 238)

  p = pyaudio.PyAudio()
  # Used for normalizing signal. If use paFloat32, then it's already -1..1.
  # Because of saving wave, paInt16 will be easier.
  MAX_y = 2.0**(p.get_sample_size(FORMAT) * 8 - 1)

  stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=BUF_SIZE)

  try:

    while True:
      t0 = clock()
      animate(lut, stream, MAX_y, state)
      t1 = clock()
      dt = t1 - t0
      delay = max(0, 1.0/FPS - dt)
      sleep(delay)
  except KeyboardInterrupt:
    print("stop")

  setleds(lut, [[0,0,0]] * 238)

  stream.stop_stream()
  stream.close()
  p.terminate()


if __name__ == '__main__':
  main()
