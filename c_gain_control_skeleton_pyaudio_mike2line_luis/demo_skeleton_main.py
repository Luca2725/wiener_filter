# ==============================================================================
# ==  2.1 IMPORTS  =============================================================
# ==============================================================================

# ~~ module access from main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
import numbers


# -- copying nparrays ----------------------------------------------------------
import copy


# ==============================================================================
# ==  3.1.1 PARAMETERS  ========================================================
# ==============================================================================

# -- manager -------------------------------------------------------------------

sleep_fraction = 0.2

# -- pyaudio -------------------------------------------------------------------

pm__chunk__sam         = 1024            # Each chunk will be 1024 samples long
pm__sample_format__int = pyaudio.paInt16 # 16 bits per sample
pm__channels__int      = 2               # Number of audio channels
pm__fs__Hz             = 44100           # Record at 44100 samples per second


# ==============================================================================
# ==  FUNCTION IMPORTS  ========================================================
# ==============================================================================

# == imports -- FBE_with_microphone ============================================
from skeleton_with_mike2line import mike_callback
from skeleton_with_mike2line import mike_queue_manager
from skeleton_with_mike2line import line_callback
from skeleton_with_mike2line import line_queue_manager
from skeleton_with_mike2line import skeleton_with_microphone
from skeleton_with_mike2line import skeleton_manager


# == imports -- high ===========================================================

# -- _______ -- ___ -- Pyaudio interface: bytes --> audio conversion -  -  -  - 
from skeleton_high_level_functions import nparray_LR_2_nparrays_L_and_R
from skeleton_high_level_functions import nparray_LR_2_nparray_L
from skeleton_high_level_functions import bytes_stereo_2_nparray_mono

# -- _______ -- ___ -- Pyaudio interface: audio --> bytes conversion -  -  -  - 
from skeleton_high_level_functions import nparray_correct_byteorder
from skeleton_high_level_functions import nparray_mono_2_bytearray_stereo


# == imports -- debug ==========================================================

from debug_funcs import PrintException



skeleton_manager( pm__chunk__sam         = pm__chunk__sam,
                  pm__sample_format__int = pm__sample_format__int,
                  pm__channels__int      = pm__channels__int,
                  pm__fs__Hz             = pm__fs__Hz,
                  sleep_fraction         = sleep_fraction
                )
