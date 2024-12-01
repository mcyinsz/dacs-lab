from env import *

import re
from argparse import ArgumentParser
from utils import execute, if_exist

def remove_clock(src_v_path, dst_v_path):
    """
        Remove clock signal from the source verilog file and save it to the destination file
    """
    with open(src_v_path, 'r') as f:
        lines = f.readlines()

    with open(dst_v_path, 'w') as f:
        for line in lines:
            if 'clock' in line:
                continue
            if 'reset' in line:
                continue
            f.write(line)


def parse_args():
    """
        Parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--hash', type=str, required=True, help='hash of the design')
    return parser.parse_args()


def main(*args, **kwargs):
    """
        Main function
    """
    args = parse_args()

    src_v_path = os.path.join(RESULT_DIR, args.hash, 'mult.v')
    dst_v_path = os.path.join(RESULT_DIR, args.hash, 'mult_no_clock.v')

    remove_clock(src_v_path, dst_v_path)

    yosys_cmd = f"cd {os.path.join(RESULT_DIR, args.hash)} && {YOSYS_BIN} -p 'read_verilog {dst_v_path}; synth -flatten; aigmap; write_aiger top.aig'"
    execute(yosys_cmd, verbose=True, wait=True)
    aig_path = os.path.join(RESULT_DIR, args.hash, 'top.aig')
    assert if_exist(aig_path)

    revsca_cmd = "export LD_LIBRARY_PATH=$HOME/.conda-yosys/lib:$LD_LIBRARY_PATH && cd %s && %s top.aig top.out -u" % (os.path.join(RESULT_DIR, args.hash), REVSCA_BIN)
    execute(revsca_cmd, verbose=True, wait=True)
    out_path = os.path.join(RESULT_DIR, args.hash, 'top.out')
    assert if_exist(out_path)


if __name__ == '__main__':
    args = parse_args()
    main(hash=args.hash)


