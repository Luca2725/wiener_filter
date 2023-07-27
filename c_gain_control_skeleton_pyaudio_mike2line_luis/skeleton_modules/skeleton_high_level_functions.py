from module_imports import *


# ==============================================================================
# ==  PYAUDIO INTERFACE  =======================================================
# ============================================================================== 

# ~~ bytes --> audio conversion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def nparray_LR_2_nparrays_L_and_R( x_t_LR
                                 ):
    if DEBUGGING:
        assert type(      x_t_LR)     == np.ndarray
        assert len(       x_t_LR)      > 1
        assert len(       x_t_LR) % 2 == 0
        assert isinstance(x_t_LR[0],     numbers.Number)

    x_t_L_idx = [i for i in range(len(x_t_LR)) if i % 2 == 0] # L channel idxes
    x_t_R_idx = [i for i in range(len(x_t_LR)) if i % 2 == 1] # R channel idxes

    x_t_L = x_t_LR[x_t_L_idx]
    x_t_R = x_t_LR[x_t_R_idx]
    
    if DEBUGGING:
        assert type(      x_t_L)    == np.ndarray
        assert len(       x_t_L)    == len(x_t_LR) / 2
        assert isinstance(x_t_L[0],    numbers.Number)
        assert            x_t_L[0]  == x_t_LR[0]
        assert type(      x_t_R)    == np.ndarray
        assert len(       x_t_R)    == len(x_t_LR) / 2
        assert isinstance(x_t_R[0],    numbers.Number)
        assert            x_t_R[0]  == x_t_LR[1]
    return (x_t_L, x_t_R)


def nparray_LR_2_nparray_L( y_new_LR
                          ):
    if DEBUGGING:
        assert type(      y_new_LR)     == np.ndarray
        assert len(       y_new_LR)      > 1
        assert len(       y_new_LR) % 2 == 0
        assert isinstance(y_new_LR[0],     numbers.Number)

    y_new_L_idx = [i for i in range(len(y_new_LR)) if i % 2 == 0] # L idxes
    y_new_L     = y_new_LR[y_new_L_idx]
    
    if DEBUGGING:
        assert type(      y_new_L)    == np.ndarray
        assert len(       y_new_L)    == len(y_new_LR) / 2
        assert isinstance(y_new_L[0],    numbers.Number)
        assert            y_new_L[0]  == y_new_LR[0]

    return y_new_L


def bytes_stereo_2_nparray_mono( frames_mike,
                                 chunk         = 1024,
                                 sample_format = pyaudio.paInt16,
                                 channels      = 2,
                                 fs            = 44100,
                               ):
    if DEBUGGING:
        assert type(frames_mike)   == bytes
        assert sum( frames_mike)   != 0
        assert type(chunk)         == int
        # ~ assert len( frames_mike)   >= chunk
        assert type(sample_format) == int
        assert type(channels)      == int
        assert type(fs)            == int

    sound = AudioSegment( data        = frames_mike,
                          sample_width= pyaudio.get_sample_size(sample_format),
                          frame_rate  = fs,
                          channels    = channels
                        )

    y_new_LR = np.array(sound.get_array_of_samples())    
    y_new_L  = nparray_LR_2_nparray_L( y_new_LR )
    y_new    = y_new_L

    if DEBUGGING:
        assert            type(y_new) ==  np.ndarray
        assert issubclass(type(y_new[0]), np.integer)

    return y_new

# ~~ audio --> bytes conversion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def nparray_correct_byteorder( nparray 
                             ):
    if DEBUGGING:
        assert type(      nparray)    == np.ndarray
        assert len(       nparray)     > 0
        assert isinstance(nparray[0],    numbers.Number)

    out = np.array([])
    if ( nparray.dtype.byteorder == '>' or 
        (nparray.dtype.byteorder == '=' and sys.byteorder == 'big')
       ):
        out = nparray.byteswap()
    else:
        out = copy.deepcopy(nparray)

    if DEBUGGING:
        assert type(      out)     == np.ndarray
        assert len(       out)     == len(nparray)
        assert isinstance(out[0],     numbers.Number)
    return out


def nparray_mono_2_bytearray_stereo( v_new,
                                     sample_format = pyaudio.paInt16
                                   ):
    if DEBUGGING:
        assert type(           v_new)         == np.ndarray
        assert len(            v_new)          > 0
        assert issubclass(type(v_new[0]),        np.integer)
        assert type(           sample_format) == int

    v_new_LR     = np.stack((v_new, v_new), axis = 1).flatten() # L --> LR
    v_new_LR_out = nparray_correct_byteorder( v_new_LR )
    frames_line  = bytearray(v_new_LR_out.ravel().view('b').data.tobytes())

    if DEBUGGING:
        assert type(frames_line) == bytearray

    return frames_line
