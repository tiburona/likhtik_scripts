from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import TEST_RUNNER_OPTS

"""
The top-level module for calling data processing functions. 
"""


def main():
    runner = Runner(config_file='/Users/katie/likhtik/IG_INED_Safety_Recall/init_config.json')
    # runner.run('make_spreadsheet', TEST_RUNNER_OPTS, path='/Users/katie/likhtik/IG_INED_Safety_Recall',
    #             filename='power_pre_pip_and_behavior')
    runner.run('plot_proportion', TEST_RUNNER_OPTS)
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
