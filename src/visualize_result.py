import os
import numpy as np
from utils.utils import (
    get_args,
    get_log,
    set_seed,
    draw_results
)

def main():
    # Get the arguments
    args = get_args()
    log_dir = args.log_dir
    print('Log directory:', log_dir)
    log_df = get_log(log_dir)
    fig = draw_results(log_df,  metrics=["bps"])
    # fig.savefig(os.path.join(log_dir, 'bps.png'))
    fig.savefig('bps.png')

if __name__ == '__main__':
    main()