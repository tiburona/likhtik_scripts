from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import TEST_RUNNER_OPTS

"""
The top-level module for calling spike processing functions. 
"""


def main():
    runner = Runner(config_file='/Users/katie/likhtik/IG_INED_Safety_Recall/init_config.json', lfp=True)
    runner.run('plot_power', TEST_RUNNER_OPTS)
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
