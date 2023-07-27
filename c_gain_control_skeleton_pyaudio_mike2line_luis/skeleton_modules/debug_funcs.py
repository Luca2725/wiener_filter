import sys
import linecache
from debugging_and_visualization import DEBUGGING, VISUALIZATION

def PrintException(): # function credit: Apogentus @ stackoverflow.com
    exc_type, exc_obj, tb = sys.exc_info()
    if type(exc_obj) == AssertionError:
        exc_obj = 'assertion error'
    f                     = tb.tb_frame
    lineno                = tb.tb_lineno
    filename              = f.f_code.co_filename
    linecache.checkcache(filename)
    filename              = filename.split('/')[-1]
    line                  = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN:\n\t{}\n\tln. {} --> {}\n\n\t{}\n'.format( filename,
                                                                     lineno,
                                                               line.strip(),
                                                                    exc_obj)
         )
