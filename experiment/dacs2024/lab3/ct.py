from env import *

import random

from design.multiplier.pp_generator import PartialProductGenerator, AndMultPPG
from design.multiplier.comp_tree import (
    CompTreeCountView, CompTreeStageView, CompTreeGraphView, 
    CompTreeCountViewMapper, SerialCompTreeCountViewMapper,
    CompTreeStageViewMapper, SortedCompTreeStageViewMapper,
    get_dadda_graph_view, get_wallace_graph_view,
)


def get_ppg(n_bit=N_BIT):
    """
        Get the partial product generator configuration
    """
    ppg = AndMultPPG(
        multiplicand_bit=n_bit,
        multiplier_bit=n_bit,
        unsigned=True,
    )
    return ppg


def get_ct(n_bit=N_BIT, c2s_iter=0, s2g_iter=0, seed=42):
    """
        Get the compressor tree configuration
    """

    rng = random.Random(seed)
    
    # We use a valid initial configuration from default tree
    count_view = get_default_ct(n_bit=N_BIT).to_stage_view().to_count_view()
    ppg = count_view.pp_gen

    # Modify the number of compressors
    # TODO: use smarter strategies for the mutation
    for i in range(c2s_iter):
        action = count_view.action_list[rng.randint(0, 3)]
        column = rng.randint(0, count_view.num_column - 1)
        new_count_array = count_view.mutate(column, action)
        if new_count_array is not None:
            count_view = CompTreeCountView(new_count_array, ppg)

    c2s_mapper = SerialCompTreeCountViewMapper()
    stage_view = c2s_mapper(count_view)

    # Modify compressor assignment to specific stages
    # TODO: use smarter strategies for the mutation
    for i in range(s2g_iter):
        column = rng.randint(0, stage_view.num_column - 1)
        src_stage = rng.randint(0, stage_view.num_stage - 1)
        dst_stage = rng.randint(0, stage_view.num_stage - 1)
        new_stage_array = stage_view.mutate(column, src_stage, dst_stage)
        if new_stage_array is not None:
            stage_view = CompTreeStageView(new_stage_array, ppg)

    s2g_mapper = SortedCompTreeStageViewMapper(pp_priority='instantiation')
    graph_view = s2g_mapper(stage_view)

    return graph_view


def get_default_ct(n_bit=N_BIT, ct_type='wallace'):
    """
        The default compressor tree is Wallace tree.
    """
    ppg = get_ppg(n_bit)

    if ct_type == 'dadda':
        ct = get_dadda_graph_view(ppg)
    elif ct_type == 'wallace':
        ct = get_wallace_graph_view(ppg)
    else:
        raise ValueError(f'Invalid CT type: {ct_type}')
    
    return ct