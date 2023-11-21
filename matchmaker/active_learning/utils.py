import os
import subprocess
import sys, traceback

os.environ['PYTHONHASHSEED'] = "42"
from transformers import logging

logging.set_verbosity_warning()

sys.path.append(os.getcwd())

def execute_and_return_run_folder(run_args):
    p = subprocess.Popen(run_args, universal_newlines=True,stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    output, err = p.communicate()
    run_folder = output.strip().split('\n')[-1]
    return run_folder

def execute_and_dont_wait(run_args):
    p = subprocess.Popen(run_args, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    output, err = p.communicate()
    run_folder = output.strip().split('\n')[-1]
    return run_folder