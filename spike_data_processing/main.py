from logger import log_directory_contents
from procs import plot_psth, plot_autocorr, plot_spectrum, make_spreadsheet, plot_proportion_score, run_post_hocs


"""
The top-level module for calling spike processing functions. Executing make() regenerates everything by default with all 
the default options, but finer-grained control is available by passing subsets of the default `levels` or procedures 
`to_run`, by calling the plot/make functions individually with non-default arguments, or changing opts in the 
opts_library.
"""


def make(levels=('group', 'animal', 'unit'), to_run=('psth', 'autocorr', 'spectrum', 'spreadsheet')):
    if 'psth' in to_run:
        plot_psth(levels)
    if 'proportion' in to_run:
        plot_proportion_score(levels=('group', 'animal'))
    if 'autocorr' in to_run:
        plot_autocorr(levels)
    if 'spectrum' in to_run:
        plot_spectrum(levels)
    if 'spreadsheet' in to_run:
        make_spreadsheet()
    if 'post_hocs' in to_run:
        run_post_hocs()


def main():

    make(to_run='post_hocs')
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
