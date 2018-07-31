from __future__ import print_function
import time
import datetime

def print_sec0(mes):
    print('##########################################################')
    print('### ', mes)

def print_sec0_end(mes=''):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++ ', mes)

def print_sec1(name):
    print('----------------------------------------------------------')
    print('--- ', name)

def print_warn(warn):
    print('!!!!! WARNING: ', warn)

def str_time(dt):
    str(datetime.timedelta(seconds=dt))

def print_proctime(dt):
    print('Processing time: ', str_time(dt))


def print2file(dict, file):
    print(dict, file=file)

