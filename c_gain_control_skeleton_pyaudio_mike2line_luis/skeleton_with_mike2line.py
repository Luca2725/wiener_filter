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


import types

# ==============================================================================
# ==  FUNCTION IMPORTS  ========================================================
# ==============================================================================

# == imports -- high ============================================================

# -- _______ -- ___ -- Pyaudio interface: bytes --> audio conversion -  -  -  - 
from skeleton_high_level_functions import nparray_LR_2_nparrays_L_and_R
from skeleton_high_level_functions import nparray_LR_2_nparray_L
from skeleton_high_level_functions import bytes_stereo_2_nparray_mono

# -- _______ -- ___ -- Pyaudio interface: audio --> bytes conversion -  -  -  - 
from skeleton_high_level_functions import nparray_correct_byteorder
from skeleton_high_level_functions import nparray_mono_2_bytearray_stereo


# == imports -- debug ==========================================================

from debug_funcs import PrintException

from numpy.fft import fft

# == FBE manager ===============================================================

def mike_callback( mike_queue  
                 ):

    if DEBUGGING:
        assert type(mike_queue) == bytearray

    def callback( in_data,     # recorded data
                  frame_count, # number of frames
                  time_info,   # dictionary
                  status       # PaCallbackFlags
                ):
        if DEBUGGING:
            assert type(in_data)     == bytes
            assert type(frame_count) == int
            assert type(time_info)   == dict
            assert type(status)      == int

        mike_queue.extend(in_data)

        out = (in_data, pyaudio.paContinue)
        if DEBUGGING:
            assert type(out)    == tuple
            assert type(out[0]) == bytes
            assert type(out[1]) == int
        return out

    if DEBUGGING:
        assert type(callback) == types.FunctionType
    return callback  


def mike_queue_manager( pls_stop_mqm,
                        mike_queue,
                        boxed_vs__y__int,
                        pm__R__sam,
                        chunk          = 1024,
                        sample_format  = pyaudio.paInt16, 
                        channels       = 2, # 1 --> mono, 2 --> stereo
                        fs             = 44100,
                        sleep_fraction = 0.2
                      ):

    if DEBUGGING:
        assert       type(pls_stop_mqm)        == threading.Event
        assert       type(mike_queue)          == bytearray
        assert       type(boxed_vs__y__int)    == list
        assert        len(boxed_vs__y__int)    == 1
        assert       type(boxed_vs__y__int[0]) == np.ndarray
        assert isinstance(boxed_vs__y__int[0][0], numbers.Number)
        assert       type(pm__R__sam)          == int
        assert       type(chunk)               == int
        assert       type(sample_format)       == int
        assert       type(channels)            == int
        assert       type(fs)                  == int
        assert       type(sleep_fraction)      == float

    sample_size        = pyaudio.get_sample_size(sample_format)
    R_samples_in_queue = pm__R__sam * channels * sample_size

    while not pls_stop_mqm.wait(0):
        try:
            if len(mike_queue) > R_samples_in_queue:

                y_new =           bytes_stereo_2_nparray_mono( 
                        bytes(mike_queue[:R_samples_in_queue]),
                                                         chunk,
                                                 sample_format,
                                                      channels,
                                                            fs)
                boxed_vs__y__int[0] = np.append(boxed_vs__y__int[0], y_new)
                del mike_queue[:R_samples_in_queue]

            else:
                time.sleep( sleep_fraction * pm__R__sam / fs ) # wait for audio
        except:
            PrintException()
            break
    
    print("MQM: stopped successfully")        
    return


def line_callback( line_queue,
                   chunk,
                   channels,
                   sample_format
                 ):

    if DEBUGGING:
        assert type(line_queue)    == bytearray
        assert type(chunk)         == int
        assert type(channels)      == int
        assert type(sample_format) == int

    def callback( in_data,     # recorded data (ie this is None bc input=False)
                  frame_count, # number of frames
                  time_info,   # dictionary
                  status       # PaCallbackFlags
                ):
        if DEBUGGING:
            assert type(in_data)     == types.NoneType
            assert type(frame_count) == int
            assert type(time_info)   == dict
            assert type(status)      == int

        out_data = bytearray()
        sz = chunk * channels * pyaudio.get_sample_size(sample_format)

        if len(line_queue) >= sz:
            out_data = line_queue[:sz]
            del line_queue[:sz]

        else:
            out_data = bytes(sz)

        out = (bytes(out_data), pyaudio.paContinue)
        if DEBUGGING:
            assert type(out)    == tuple
            assert type(out[0]) == bytes
            assert type(out[1]) == int
        return out

    if DEBUGGING:
        assert type(callback) == types.FunctionType
    return callback


def line_queue_manager( pls_stop_lqm,
                        line_queue,
                        boxed_vs__v__int,
                        pm__R__sam,
                        sample_format  = pyaudio.paInt16, 
                        channels       = 2, # 1 --> mono, 2 --> stereo
                        fs             = 44100,
                        sleep_fraction = 0.2
                      ):

    if DEBUGGING:
        assert       type(pls_stop_lqm)        == threading.Event
        assert       type(line_queue)          == bytearray
        assert       type(boxed_vs__v__int)    == list
        assert        len(boxed_vs__v__int)    == 1
        assert       type(boxed_vs__v__int[0]) == np.ndarray
        assert isinstance(boxed_vs__v__int[0][0], numbers.Number)
        assert       type(pm__R__sam)          == int
        assert       type(sample_format)       == int
        assert       type(channels)            == int
        assert       type(fs)                  == int
        assert       type(sleep_fraction)      == float

    sample_size        = pyaudio.get_sample_size(sample_format)
    R_samples_in_queue = pm__R__sam * channels * sample_size

    while not pls_stop_lqm.wait(0):
        try:
            if len(boxed_vs__v__int[0]) > pm__R__sam:

                v_new           = boxed_vs__v__int[0][:pm__R__sam]
                v_new_bytearray = nparray_mono_2_bytearray_stereo( v_new,
                                                                   sample_format
                                                                 )
                line_queue.extend(v_new_bytearray)
                boxed_vs__v__int[0] = boxed_vs__v__int[0][pm__R__sam:]

            else:
                time.sleep( sleep_fraction * pm__R__sam / fs ) # wait for audio
        except:
                PrintException()
                break
    
    print("LQM: stopped successfully")        
    return


def skeleton_with_microphone( pls_stop_fbe,
                              boxed_vs__y__int,
                              boxed_vs__v__int,
                              pm__chunk__sam = 1024
                            ):

    if DEBUGGING:
        assert       type(pls_stop_fbe)           == threading.Event
        assert       type(boxed_vs__y__int)       == list
        assert        len(boxed_vs__y__int)       == 1
        assert       type(boxed_vs__y__int[0])    == np.ndarray
        assert isinstance(boxed_vs__y__int[0][0],    numbers.Number)
        assert       type(boxed_vs__v__int)       == list
        assert        len(boxed_vs__v__int)       == 1
        assert       type(boxed_vs__v__int[0])    == np.ndarray
        assert isinstance(boxed_vs__v__int[0][0],    numbers.Number)
        assert       type(pm__chunk__sam)         == int

    n_now   = 0
    n_now_y = 0
    n_now_v = 0
    monad_v = np.zeros(pm__chunk__sam).astype('int16')

    while not pls_stop_fbe.wait(0):
        try:
            if ( len(boxed_vs__y__int[0]) > pm__chunk__sam and
                 len(boxed_vs__y__int[0]) > n_now_y        ):

                n_now   += 1
                n_now_y += 1
                n_now_v += 1

                if n_now % pm__chunk__sam == 0:
                    boxed_vs__y__int[0] = boxed_vs__y__int[0][pm__chunk__sam:]
                    boxed_vs__v__int[0] = np.append(boxed_vs__v__int[0],monad_v)
                    
                    monad_v           = np.zeros(pm__chunk__sam).astype('int16')
                    '''
                    COMECEI AQUI
                    Configurações do algoritmo esTSNR
                    '''
                    
                    '''
                    garantimos que o tamanho da janela tem o mesmo tamanho que o segmento de audio
                    '''
                    wl = pm__chunk__sam
                    #wl = int(0.025 * fs)  # Comprimento da janela é de 25 ms
                    nfft = 2 * wl  # Tamanho do FFT é o dobro do comprimento da janela
                    hanwin = np.hanning(wl)  # Janela de Hann
                    Is = 10 * wl  # Tamanho da soma dos quadradros de janelas consecutivas
                    
                    # Aplicação do algoritmo esTSNR no chunk de áudio
                    nsum = np.zeros(nfft, dtype=np.float64)
                    count = 0
                    
                    for m in range(0, Is - wl, wl):
                        '''
                        Isso irá interromper o loop caso não haja amostras suficientes para 
                        formar o segmento de áudio de tamanho wl. Dessa forma, o código não 
                        tentará realizar a multiplicação entre arrays de tamanhos incompatíveis.
                        '''
                        if m + wl > len(monad_v):
                            break
                        nwin = monad_v[m : m + wl] * hanwin
                        nsum = nsum + np.abs(fft(nwin, nfft)) ** 2
                        count += 1
                    
                    d = nsum / count
                    d = d[:, np.newaxis]
                    
                    SP = 0.4
                    normFactor = 1 / SP
                    overlap = int((1 - SP) * wl)  # Overlap entre frames sucessivos
                    offset = wl - overlap
                    max_m = int((pm__chunk__sam - nfft) / offset)
                    
                    zvector = np.zeros((nfft, 1))
                    oldmag = np.zeros((nfft, 1))
                    news = np.zeros((pm__chunk__sam, 1))
                    
                    alpha = 0.99
                    
                    for m in range(max_m):
                        begin = m * offset + 1
                        iend = m * offset + wl
                        speech = monad_v[begin:iend + 1]
                        winy = hanwin * speech
                        ffty = np.fft.fft(winy, nfft)
                        phasey = np.angle(ffty)
                    
                        postsnr = (np.power(oldmag, 2) / d) - 1
                        postsnr = np.maximum(postsnr, 0.1)
                    
                        eta = alpha * np.divide(np.square(oldmag), d) + postsnr * (1 - alpha)
                    
                        newmag = np.multiply((eta / (eta + 1)), np.abs(ffty)) #np.abs(ffty) = magnitude
                        tsnr = (np.power(newmag, 2) / d)
                        Gtsnr = tsnr / (tsnr + 1)
                    
                        Gtsnr = np.maximum(Gtsnr, 0.15)
                        
                        #-----------------------#
                        #------gain control ----#
                        ConstraintInLength = nfft // 2
                        meanGain = np.mean(np.power(Gtsnr,2))
                        nfft1 = len(Gtsnr)      #nfft1 é definido como o comprimento de Gtsnr
                        L2 = ConstraintInLength     #L2 é o comprimento da restrição.
                        win = np.hamming(L2)        #A janela de Hamming win é criada com base em L2
                        ImpulseR = np.real(np.fft.ifft(Gtsnr))  #A resposta ao impulso real ImpulseR é calculada através da transformada inversa de Fourier do ganho Gtsnr
                    
                        #é realizado um processo de janela deslizante na resposta ao impulso ImpulseR, multiplicando as partes correspondentes pela janela de Hamming.
                        #A resposta ao impulso resultante é armazenada em ImpulseR2.
                        ImpulseR2 = np.concatenate([np.multiply(ImpulseR[:L2//2],win[L2//2:]), np.zeros(nfft1 - L2), np.multiply(ImpulseR[nfft1 - L2//2:],win[:L2//2])])
                        NewGain = np.abs(np.fft.fft(ImpulseR2, nfft1))  #é aplicada a transformada de Fourier para obter o novo ganho NewGain
                        meanNewGain = np.mean(np.power(NewGain,2))      # O valor médio do novo ganho é calculado em meanNewGain.
                        NewGain = NewGain * np.sqrt(meanGain / meanNewGain) 
                        #-----------------------#
                        #------gain control ----#
                        Gtsnr = NewGain
                    
                    
                        newmag = np.multiply(Gtsnr, np.abs(ffty))
                    
                        ffty = newmag * np.exp(1j * phasey)
                        newmag1 = newmag[:, np.newaxis]
                    
                        oldmag = np.abs(newmag1)
                        soma = np.real(np.fft.ifft(ffty, nfft))
                        soma1 = soma[:, np.newaxis]
                        news[begin:begin + nfft] += soma1 / normFactor
                    
                    # Atualização do chunk de áudio com o resultado do algoritmo esTSNR
                    monad_v = news.astype('int16')

                    '''
                    O comprimento da janela wl é definido como o tamanho do chunk de áudio pm__chunk__sam. 
                    O tamanho da FFT (nfft) é definido como o dobro do comprimento da janela. 
                    A janela de Hann é criada com comprimento wl. O valor Is é o tamanho da soma dos quadrados de
                     janelas consecutivas e é definido como 10 vezes o comprimento da janela.
                    
                     o algoritmo percorre o chunk de áudio em janelas consecutivas de tamanho wl com um 
                     passo de wl. A cada iteração, uma janela (nwin) é extraída do chunk de áudio monad_v e
                     multiplicada pela janela de Hann. O resultado é a soma dos quadrados da FFT aplicada à janela.
                     O valor resultante é armazenado em nsum, e o contador count é incrementado. 

                     _No meio dos ciclos, essas linhas definem variáveis adicionais utilizadas no algoritmo. SP representa a taxa de 
                     deslocamento entre frames sucessivos. normFactor é o fator de normalização usado para ajustar 
                     o som resultante. overlap é o tamanho da sobreposição entre frames sucessivos, calculado com
                     base na taxa de deslocamento. offset é o deslocamento entre inícios de janelas consecutivas.
                     max_m é o número máximo de iterações a serem executadas no loop seguinte. zvector, oldmag e news
                     são arrays de zeros usados para armazenar informações intermediárias durante o processamento. alpha
                     é um parâmetro de controle usado no cálculo do ganho do esTSNR.
                    
                     o algoritmo percorre o chunk de áudio em frames consecutivos com um deslocamento definido por offset
                     . Para cada frame, uma janela (winy) é extraída do chunk de áudio, multiplicada pela janela de Hann e
                     tem a FFT aplicada. O espectro resultante é usado para calcular o ganho (Gtsnr) com base nas
                     estimativas de potência de ruído e de sinal. Esse ganho é aplicado ao espectro original para obter 
                     um novo espectro (newmag).
                     A parte da função do gain control é responsável por ajustar o ganho do sinal de áudio processado. 
                     Após calcular a relação sinal-ruído de limiar (Gtsnr), é realizada a função de controle de ganho para normalizar 
                     o ganho do sinal. Dessa forma, garante-se que o sinal resultante tenha um nível de volume adequado, evitando
                     distorções causadas por amplificação excessiva ou atenuação insuficiente.
                     Em seguida, o novo espectro é transformado de volta para o domínio do 
                     tempo por meio da IFFT, e o resultado é adicionado ao array news.
                    
                    No final do código, o chunk de áudio monad_v é atualizado com o resultado do algoritmo esTSNR
                     armazenado em news.
                    '''
                    n_now_y -= pm__chunk__sam
                    n_now_v -= pm__chunk__sam
            
            
            monad_v[n_now_v] = boxed_vs__y__int[0][n_now_y] #original
        except:
            PrintException()
            break
    
    print("FBE_with_microphone: successfully returning")
    return
    
    print("FBE_with_microphone: successfully returning")
    return



def skeleton_manager( pm__chunk__sam         = 1024,
                      pm__sample_format__int = pyaudio.paInt16,
                      pm__channels__int      = 2,
                      pm__fs__Hz             = 44100,
                      sleep_fraction         = 0.2
                    ):

    if DEBUGGING:
        assert type(pm__chunk__sam)         == int
        assert type(pm__sample_format__int) == int
        assert type(pm__channels__int)      == int
        assert type(pm__fs__Hz)             == int
        assert type(sleep_fraction)         == float

    pa            = pyaudio.PyAudio() 
    chunk         = pm__chunk__sam
    sample_format = pm__sample_format__int
    channels      = pm__channels__int
    fs            = pm__fs__Hz

    # -- 1. Instantiate FBE-shared variables -----------------------------------

    boxed_vs__y__int = [np.array([0]).astype('int16')] # this is to prevent
    boxed_vs__v__int = [np.array([0]).astype('int16')] # initialization errors

    # -- 2. Prepare mike audio environment -------------------------------------

    print("MGR: setting up mike audio environment")
    mike_queue  = bytearray()
    mike_stream = pa.open( format            = sample_format,
                           channels          = channels,
                           rate              = fs,
                           input             = True,
                           frames_per_buffer = chunk,
                           stream_callback   = mike_callback( mike_queue )
                         )

    pls_stop_mqm = threading.Event()
    mqm_thread   = threading.Thread( target = mike_queue_manager,
                                     args   = ( pls_stop_mqm,
                                                mike_queue,
                                                boxed_vs__y__int,
                                                pm__chunk__sam,
                                                chunk,
                                                sample_format, 
                                                channels,
                                                fs,
                                                sleep_fraction
                                              )
                                   )

    # -- 3. Prepare line audio environment -------------------------------------

    print("MGR: setting up line audio environment")
    line_queue  = bytearray()
    line_stream = pa.open( format            = sample_format,
                           channels          = channels,
                           rate              = fs,
                           output            = True,
                           frames_per_buffer = chunk,
                           stream_callback   = line_callback( line_queue,
                                                              chunk,
                                                              channels,
                                                              sample_format
                                                            )
                         )

    pls_stop_lqm = threading.Event()
    lqm_thread   = threading.Thread( target = line_queue_manager,
                                     args   = ( pls_stop_lqm,
                                                line_queue,
                                                boxed_vs__v__int,
                                                pm__chunk__sam,
                                                sample_format, 
                                                channels,
                                                fs,
                                                sleep_fraction
                                              )
                                   )

    # -- 4. Prepare FBE --------------------------------------------------------

    print("MGR: preparing skeleton")
    pls_stop_fbe = threading.Event()
    fbe_thread   = threading.Thread( target = skeleton_with_microphone,
                                     args   = ( pls_stop_fbe,
                                                boxed_vs__y__int,
                                                boxed_vs__v__int,
                                                pm__chunk__sam
                                              )
                                   )

    # -- 5. Start all threads --------------------------------------------------

    print("MGR: starting threads")
    mqm_thread.start()
    lqm_thread.start()
    fbe_thread.start()

    # -- 6. Keep running until I want it to stop -------------------------------

    print("MGR: running. Press Ctrl + c to stop")
    while mike_stream.is_active():
        try:
            if DEBUGGING:
                assert line_stream.is_active()
                assert mqm_thread.is_alive()
                assert lqm_thread.is_alive()
                assert fbe_thread.is_alive()
            time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nMGR: stopping...")
            mike_stream.stop_stream()
            break

    # -- 7. Safely stop all threads --------------------------------------------

    while mike_stream.is_active():
        time.sleep(0.1)
    print("MGR: mike stream stopped. Stopping mike queue manager")
    pls_stop_mqm.set()

    while mqm_thread.is_alive():
        time.sleep(0.1)
    print("MGR: mike queue manager stopped. Stopping FBE")
    pls_stop_fbe.set()

    while fbe_thread.is_alive():
        time.sleep(0.1)
    print("MGR: FBE stopped. Stopping line queue manager")
    pls_stop_lqm.set()

    while lqm_thread.is_alive():
        time.sleep(0.1)
    print("MGR: line queue manager stopped. Stopping line stream")
    line_stream.stop_stream()

    while mike_stream.is_active():
        time.sleep(0.1)
    print("MGR: line stream stopped. Terminating streams and threads")

    # -- 8. Terminate streams and threads --------------------------------------

    pa.terminate()
    mqm_thread.join()
    lqm_thread.join()
    fbe_thread.join()
    print("MGR: done!")
    return





