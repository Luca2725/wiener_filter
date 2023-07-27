import sys
sys.path.append('./skeleton_modules')

# -- audio manipulation --------------------------------------------------------
import numpy as np
import math
from pydub import AudioSegment


# -- record / playback ---------------------------------------------------------
import pyaudio
import time


# -- DCT -----------------------------------------------------------------------
# ~ import scipy as sp


# -- sidechain -----------------------------------------------------------------
import threading


# -- visualization -------------------------------------------------------------
# ~ from debugging_and_visualization import VISUALIZATION
# ~ import matplotlib.pyplot as plt 
# ~ import scipy.signal as signal
# ~ from   scipy.fft import fft, fftfreq, fftshift, dct
# ~ import multiprocessing as mp

# -- debugging -----------------------------------------------------------------
from debugging_and_visualization import DEBUGGING
from debug_funcs import PrintException
import numbers


# -- copying nparrays ----------------------------------------------------------
import copy
