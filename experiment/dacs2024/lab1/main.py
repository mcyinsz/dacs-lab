from env import *

import json
import random

from design.adder.add_config import PPAdderConfig
from design.adder.add_baseline import get_Sklansky_adder, get_KoggeStone_adder, get_BrentKung_adder
from tech.asap7 import Asap7Library
from flow.genus_innovus import GenusInnovusFlow


def get_design_config(adder_type='sklansky', input_bit=64) -> dict:
    """
        get the adder design configuration with specific type and bit
    """

    if adder_type == 'sklansky':
        adder = get_Sklansky_adder(input_bit)
    elif adder_type == 'koggestone':
        adder = get_KoggeStone_adder(input_bit)
    elif adder_type == 'brentkung':
        adder = get_BrentKung_adder(input_bit)
    else:  # fall back to ripple carry adder
        adder = PPAdderConfig(input_bit)

    ###########################################################################
    # TODO: modify the default adder structure with 'add' or 'delete' operation
    ###########################################################################
    # adder = modify_adder_design(adder)

    save_dir = os.path.join(RESULT_DIR, adder.hash)
    adder.dump_data(save_dir)

    design_config = {
        'verilog_files': [os.path.join(save_dir, 'ppadder.v')],
        'top_module': 'PPAdder',
        'clk_name': 'clock',
        'clk_port_name': 'clock',
    }

    return design_config


def modify_adder_design(adder: PPAdderConfig) -> PPAdderConfig:
    """
        An exmaple code to modify the adder structrue with random action
        Reference PrefixRL: https://ieeexplore.ieee.org/abstract/document/9586094/
    """

    n_bit = adder.N
    rnd = random.random()
    flag = False  # if the action is effective

    while not flag:
        msb = random.randint(0, n_bit-1)
        lsb = random.randint(0, msb)
        
        if rnd < 0.5:
            flag = adder.add(msb, lsb)
        else:
            flag = adder.delete(msb, lsb)

    return adder


def get_tech_config() -> dict:
    """
        get standard cell library configuration
    """
    return Asap7Library(ASAP7_ROOT).to_dict()


def get_syn_options() -> dict:
    """
        Use Cadence Genus for logic synthesis and set the tool options
    """
    syn_configs = {
        'genus_bin': GENUS_BIN,
        'max_threads': 1,
        'steps': ['syn', 'report'],
        
        ###########################################################################
        # TODO: modify the following synthesis options
        ###########################################################################

        # target timing: float
        'clk_period_ns': 0.0,

        # generic logical synthesis effort: [low/medium/high]
        'syn_generic_effort': 'medium',

        # technology mapping synthesis effort: [low/medium/high]
        'syn_map_effort': 'high',

        # post-synthesis optimization effort: [None/low/medium/high]
        'syn_opt_effort': None,

        # fanout constraint: int
        "max_fanout": None,

        # transition constraint: float
        "max_transition_ns": None,

        # capacitance constraint: float
        "max_capacitance_ff": None,
    }

    return syn_configs


def get_pnr_options() -> dict:
    """
        Use Cadence Innovus for physical design and set the tool options
    """
    pnr_configs = {
        'innovus_bin': INNOVUS_BIN,
        'max_threads': 1,
        'steps': [
            'init',
            'floorplan',
            'powerplan',
            'placement',
            # 'cts',      # the adder module is purely combinatorial and does not require CTS stage
            'routing',
        ],
        'runmode': 'fast',  # use 'skip' if you don't need to run physical design, useful for debugging
        
        ###########################################################################
        # TODO: modify the following synthesis options
        ###########################################################################

        # placement floorplan utilization: float, 0~1
        'place_utilization': 0.6,

        # specify max distance (in micron) for refinePlace ECO mode: float, min=0, max=9999
        'place_detail_eco_max_distance':  10.0,

        # select instance priority for refinePlace ECO mode: [placed/fixed/eco]
        'place_detail_eco_priority_insts':  'placed',

        # detail placement considers optimizing activity power: [true/false]
        'place_detail_activity_power_driven':  'false',

        # wire length optimization effort: [low/medium/high]
        'place_detail_wire_length_opt_effort':  'medium',

        # minimum gap between instances (unit sites): int, default=0
        'place_detail_legalization_inst_gap':  2,

        # Placement will (temporarily) block channels between areas with limited routing capacity: [none/soft/partial]
        'place_global_auto_blockage_in_channel':  'none',

        # identifies and constrains power-critical nets to reduce switching power: [true/false]
        'place_global_activity_power_driven':  'false',

        # power driven effort: [standard/high]
        'place_global_activity_power_driven_effort':  'standard',

        # clock power driven: [true/false]
        'place_global_clock_power_driven':  'true',

        # clock power driven effort: [low/standard/high]
        'place_global_clock_power_driven_effort':  'low',

        # level of effort for timing driven global placer: [meduim/high]
        'place_global_timing_effort':  'medium',

        # level of effort for congestion driven global placer: [low/medium/high/extreme/auto]
        'place_global_cong_effort':  'auto',

        # placement strives to not let density exceed given value, in any part of design: float, default=-1 for no constraint
        # you can set to 0~1
        'place_global_max_density':  -1.00,

        # find better placement for clock gating elements towards the center of gravity for fanout: [true/false]
        'place_global_clock_gate_aware':  'true',

        # enable even cell distribution for designs with less than 70% utilization: [true/false]
        'place_global_uniform_density':  'false',

    }

    return pnr_configs


def main():
    """
        Run the complete EDA flow for final PPA
    """

    design_config = get_design_config()
    tech_config = get_tech_config()
    syn_options = get_syn_options()
    pnr_options = get_pnr_options()

    # hint: you need to modify tool rundir if you change the default tool options!
    rundir = os.path.dirname(design_config['verilog_files'][0])

    flow = GenusInnovusFlow(
        design_config=design_config,
        tech_config=tech_config,
        syn_options=syn_options,
        pnr_options=pnr_options,
        rundir=rundir,
    )

    result = flow.run()

    with open(os.path.join(rundir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    print(result)


if __name__ == '__main__':
    main()