from env import *

import json
from time import time
from multiprocessing import Pool
from functools import partial
from argparse import ArgumentParser

from ct import get_ct
from cpa import get_cpa
from design.multiplier.top_module import MultTop
from tool import get_syn_options, get_pnr_options
from tech.asap7 import Asap7Library
from flow.genus_innovus import GenusInnovusFlow
from utils import create_hash, mkdir


def get_tech_config() -> dict:
    """
        get standard cell library configuration
    """
    return Asap7Library(ASAP7_ROOT).to_dict()


def get_rundir(mult: MultTop, syn_options: dict, pnr_options: dict) -> str:
    """
        Create a unique directory for the current run
    """
    ct_hash = mult.comp_tree.hash
    cpa_hash = mult.final_adder.hash
    syn_str = '_'.join([f'{k}={v}' for k, v in syn_options.items()])
    pnr_str = '_'.join([f'{k}={v}' for k, v in pnr_options.items()])
    rundir = os.path.join(RESULT_DIR, create_hash(ct_hash + cpa_hash + syn_str + pnr_str))
    return rundir


def main(*args, **kwargs):
    """
        Run the complete EDA flow for final PPA
    """
    seed = 42 if kwargs.get('debug', False) else int(time() * 1e12)
    ct_c2s_iter = 0 if kwargs.get('debug', False) else 100
    ct_s2g_iter = 0 if kwargs.get('debug', False) else 100
    cpa_iter    = 0 if kwargs.get('debug', False) else 100

    mult = MultTop(
        comp_tree=get_ct(
            c2s_iter=ct_c2s_iter,
            s2g_iter=ct_s2g_iter,
            seed=seed
        ),
        final_adder=get_cpa(
            iter=cpa_iter,
            seed=seed
        ),
    )
    tech_config = get_tech_config()
    syn_options = get_syn_options(seed=None if kwargs.get('debug', False) else seed)
    pnr_options = get_pnr_options(seed=None if kwargs.get('debug', False) else seed)

    rundir = get_rundir(mult, syn_options, pnr_options)
    mkdir(rundir)

    verilog_path = os.path.join(rundir, 'mult.v')
    with open(verilog_path, 'w') as f:
        f.write(mult.generate_verilog(top_module_name='MultTop'))

    design_config = {
        'verilog_files': [verilog_path],
        'top_module': 'MultTop',
        'clk_name': 'clock',
        'clk_port_name': 'clock',
    }

    flow = GenusInnovusFlow(
        design_config=design_config,
        tech_config=tech_config,
        syn_options=syn_options,
        pnr_options=pnr_options,
        rundir=rundir,
    )

    flow.run()

    result = {
        'delay': flow.get_timing('postRoute'),
        'area':  flow.get_floorplan_area(),
        'drv':   flow.get_drv(),
    }

    with open(os.path.join(rundir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    print(result)


def parse_args():
    """
        Parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument('-n', type=int, default=1, help='number of runs')
    parser.add_argument('-w', type=int, default=1, help='number of workers')
    parser.add_argument('--debug', action='store_true', default=False, help='enable debug mode')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with Pool(args.w) as p:
        p.map(partial(main, debug=args.debug), range(args.n))