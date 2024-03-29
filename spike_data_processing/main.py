from utils import log_directory_contents
from procs import *

"""
The top-level module for calling spike processing functions. 
"""


def main():

    make_lfp_firing_rate_spreadsheet()  # An example.  See what else is available in the procs module.
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
