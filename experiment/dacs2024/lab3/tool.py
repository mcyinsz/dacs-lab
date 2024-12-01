from env import *
from space import *

def get_syn_options(seed=42) -> dict:
    """
        Generate synthesis options
    """
    syn_options = {
        'genus_bin': GENUS_BIN,
        'max_threads': 1,  # you can use multiple CPU core to speed up synthesis
        'steps': ['syn', 'report'],

        'clk_period_ns': 1.0,
        'syn_generic_effort': 'medium',
        'syn_map_effort': 'high',
        'syn_opt_effort': None,
        "max_fanout": None,
        "max_transition_ns": None,
        "max_capacitance_ff": None,

    }

    # TODO: use better strategies to sample the synthesis options
    if seed is not None:
        genus_config_space = BaseDesignSpace(CADENCE_GENUS_DESIGN_SPACE)
        syn_options.update(genus_config_space.generate_design_point_by_random(seed))

    return syn_options


def get_pnr_options(seed=42) -> dict:
    """
        Generate place-and-route options
    """
    pnr_options = {
        'innovus_bin': INNOVUS_BIN,
        'max_threads': 1,  # you can use multiple CPU core to speed up PnR
        'steps': [
            'init',
            'floorplan',
            'powerplan',
            'placement',
            'routing',
            'floorplan_area',
            'run_drv',
        ],

        'place_utilization': 0.5,
        'place_detail_eco_priority_insts': 'placed',
        'place_detail_activity_power_driven': 'true',
        'place_detail_wire_length_opt_effort': 'medium',
        'place_global_auto_blockage_in_channel': 'none',
        'place_global_activity_power_driven': 'true',
        'place_global_activity_power_driven_effort': 'standard',
        'place_global_timing_effort': 'medium',
        'place_global_cong_effort': 'medium',
        'place_global_uniform_density': 'true',
    }

    # TODO: use better strategies to sample the PnR options
    if seed is not None:
        innovus_config_space = BaseDesignSpace(CADENCE_INNOVUS_DESIGN_SPACE)
        pnr_options.update(innovus_config_space.generate_design_point_by_random(seed))

    return pnr_options




CADENCE_GENUS_DESIGN_SPACE = [
    {
        'name': 'clk_period_ns',
        'type': 'categorical',
        'choices': [0.5, 1.0, 1.5, 2.0],
    },
    {
        'name': 'syn_generic_effort',
        'type': 'categorical',
        'choices': ['low', 'medium', 'high'],
    },
    {
        'name': 'syn_map_effort',
        'type': 'categorical',
        'choices': ['low', 'medium', 'high'],
    },
    {
        'name': 'syn_opt_effort',
        'type': 'categorical',
        'choices': [None, 'low', 'medium', 'high'],
    },
    # {
    #     'name': 'max_fanout',
    #     'type': 'categorical',
    #     'choices': [None, 10, 20, 30],
    # },
    # {
    #     'name': 'max_transition_ns',
    #     'type': 'categorical',
    #     'choices': [None, 0.1, 0.2, 0.3],
    # },
    # {
    #     'name': 'max_capacitance_ff',
    #     'type': 'categorical',
    #     'choices': [None, 0.1, 0.2, 0.3],
    # }
]


CADENCE_INNOVUS_DESIGN_SPACE = [
    {
        'name': 'place_utilization',
        'type': 'categorical',
        'choices': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    },
    {
        'name': 'place_detail_eco_priority_insts',
        'type': 'categorical',
        'choices': ['placed', 'fixed', 'eco'],
    },
    {
        'name': 'place_detail_activity_power_driven',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
    {
        'name': 'place_detail_wire_length_opt_effort',
        'type': 'categorical',
        'choices': ['none', 'medium', 'high'],
    },
    {
        'name': 'place_global_auto_blockage_in_channel',
        'type': 'categorical',
        'choices': ['none', 'soft', 'partial'],
    },
    {
        'name': 'place_global_activity_power_driven',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
    {
        'name': 'place_global_activity_power_driven_effort',
        'type': 'categorical',
        'choices': ['standard', 'high'],
    },
    # {
    #     'name': 'place_global_clock_power_driven',
    #     'type': 'categorical',
    #     'choices': ['true', 'false'],
    # },
    # {
    #     'name': 'place_global_clock_power_driven_effort',
    #     'type': 'categorical',
    #     'choices': ['low', 'standard', 'high'],
    # },
    {
        'name': 'place_global_timing_effort',
        'type': 'categorical',
        'choices': ['medium', 'high'],
    },
    {
        'name': 'place_global_cong_effort',
        'type': 'categorical',
        'choices': ['low', 'medium', 'high', 'auto'],
    },
    # {
    #     'name': 'place_global_clock_gate_aware',
    #     'type': 'categorical',
    #     'choices': ['true', 'false'],
    # },
    {
        'name': 'place_global_uniform_density',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
]