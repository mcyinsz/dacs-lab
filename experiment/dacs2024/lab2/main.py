from env import *

import json
import random
from time import time

from metric import MetricParser
from space import BaseDesignSpace, CADENCE_GENUS_DESIGN_SPACE, CADENCE_INNOVUS_DESIGN_SPACE
from tech.asap7 import Asap7Library
from flow.genus_innovus import GenusInnovusFlow
from utils import create_hash


def get_design_config() -> dict:
    """
        get the adder design configuration with specific type and bit
    """
    verilog_files = [
        "ibex_alu.v", 
        "ibex_branch_predict.v", 
        "ibex_compressed_decoder.v", 
        "ibex_controller.v", 
        "ibex_core.v", 
        "ibex_counter.v", 
        "ibex_cs_registers.v", 
        "ibex_csr.v", 
        "ibex_decoder.v", 
        "ibex_dummy_instr.v", 
        "ibex_ex_block.v", 
        "ibex_fetch_fifo.v", 
        "ibex_icache.v", 
        "ibex_id_stage.v", 
        "ibex_if_stage.v", 
        "ibex_load_store_unit.v", 
        "ibex_multdiv_fast.v", 
        "ibex_multdiv_slow.v", 
        "ibex_pmp.v", 
        "ibex_prefetch_buffer.v", 
        "ibex_register_file_ff.v", 
        "ibex_register_file_fpga.v", 
        "ibex_register_file_latch.v", 
        "ibex_wb_stage.v", 
        "prim_badbit_ram_1p.v", 
        "prim_clock_gating.v", 
        "prim_generic_clock_gating.v", 
        "prim_generic_ram_1p.v", 
        "prim_lfsr.v", 
        "prim_ram_1p.v", 
        "prim_secded_28_22_dec.v", 
        "prim_secded_28_22_enc.v", 
        "prim_secded_39_32_dec.v", 
        "prim_secded_39_32_enc.v", 
        "prim_secded_72_64_dec.v", 
        "prim_secded_72_64_enc.v", 
        "prim_xilinx_clock_gating.v",
    ]
    verilog_files = [os.path.join(LAB2_ROOT, 'ibex', f) for f in verilog_files]

    design_config = {
        'verilog_files': verilog_files,
        'top_module': 'ibex_core',
        'clk_name': 'core_clock',
        'clk_port_name': 'clk_i',
    }

    return design_config


def get_test_design_config() -> dict:
    design_config = {
        'verilog_files': [os.path.join(CLDSE_ROOT, 'design/example/gcd/gcd.v')],
        'top_module': 'gcd',
        'clk_name': 'clk',
        'clk_port_name': 'clk',
    }

    return design_config


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
        'clk_period_ns': 10.0,

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

    # TODO: we randomly sample synthesis options here
    # you can use other methods to sample the synthesis options
    genus_config_space = BaseDesignSpace(CADENCE_GENUS_DESIGN_SPACE)
    syn_configs.update(genus_config_space.generate_design_point_by_random(seed=int(time())))

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
            'cts',
            'routing',
            'extract_rc',
            'chipdone_slack',
            'chipdone_static_power',
            'floorplan_area',
            'run_drv',
        ],
        'runmode': 'fast',  # use 'skip' if you don't need to run physical design, useful for debugging
        
        ###########################################################################
        # TODO: modify the following synthesis options
        ###########################################################################

        # placement floorplan utilization: float, 0~1
        'place_utilization': 0.4,

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

    # TODO: we randomly sample pnr options here
    # you can use other methods to sample the pnr options
    innovus_config_space = BaseDesignSpace(CADENCE_INNOVUS_DESIGN_SPACE)
    pnr_configs.update(innovus_config_space.generate_design_point_by_random(seed=int(time())))

    return pnr_configs


def get_rundir(syn_options: dict, pnr_options: dict) -> str:
    """
        Create a unique directory for the current run
    """
    syn_str = '_'.join([f'{k}={v}' for k, v in syn_options.items()])
    pnr_str = '_'.join([f'{k}={v}' for k, v in pnr_options.items()])
    rundir = os.path.join(RESULT_DIR, create_hash(syn_str + pnr_str))
    return rundir


def main():
    """
        Run the complete EDA flow for final PPA
    """

    design_config = get_design_config()
    tech_config = get_tech_config()
    syn_options = get_syn_options()
    pnr_options = get_pnr_options()

    with open(os.path.join(rundir, 'config.json'), 'w') as f:
        json.dump({
            'syn_options': syn_options,
            'pnr_options': pnr_options,
        }, f, indent=4)

    rundir = get_rundir(syn_options, pnr_options)

    flow = GenusInnovusFlow(
        design_config=design_config,
        tech_config=tech_config,
        syn_options=syn_options,
        pnr_options=pnr_options,
        rundir=rundir,
    )

    flow.run()

    result = MetricParser(rundir).generate_report()
    result['clk_period_ns'] = syn_options['clk_period_ns']

    with open(os.path.join(rundir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    print(result)


if __name__ == '__main__':
    main()