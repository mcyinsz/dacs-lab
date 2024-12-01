import math
import numpy as np
import networkx as nx
from typing import List
from dataclasses import asdict

from .graph_utils import Compressor, CompressorType, PinType, Connection
from .graph_view import CompTreeGraphView
from design.multiplier.pp_generator import PartialProductGenerator

from utils import info


def pp_list_to_pp_array(pp_list: List[List[Connection]]) -> List[List[Connection]]:
    """
        Convert pp list into an parallelogram-shape pp array
    """
    pp_vec = np.array([len(l) for l in pp_list])
    num_row = np.max(pp_vec)
    num_col = len(pp_list)
    max_pp_col = num_col - 1 - np.argmax(pp_vec[::-1])  # debug: find the last argmax
    pp_array = [[None for col in range(num_col)] for row in range(num_row)]

    for col in range(num_col):
        cur_ppcnt = len(pp_list[col])
        row_range = range(cur_ppcnt) if (col < max_pp_col) else range(num_row-cur_ppcnt, num_row)
        for i, row in enumerate(row_range):
            pp_array[row][col] = pp_list[col][i]

    return pp_array


def pp_array_to_pp_list(pp_array: List[List[Connection]]) -> List[List[Connection]]:
    """
        Convert parallelograpm-shape pp array back to pp list
    """
    num_row = len(pp_array)
    num_col = len(pp_array[0])
    pp_list = [[] for col in range(num_col)]

    for row in range(num_row):
        for col in range(num_col):
            if pp_array[row][col] is not None:
                pp_list[col].append(pp_array[row][col])

    return pp_list


def get_maximum_height_sequence(num_stage):
    """
        Get maximum height s[i] for each stage i
    """
    seq = [2]
    for i in range(1, num_stage):
        seq.append(math.floor(1.5*seq[i-1]))
    return seq


def get_wallace_graph_view(pp_gen: PartialProductGenerator) -> CompTreeGraphView:
    """
        Generate a DAG representing wallace multiplier graph
    """

    global_node_list = []
    global_pp_list = []

    # initial pp list
    init_ppcnt_list = pp_gen.get_init_ppcnt_list()
    num_col = pp_gen.num_column
    pp_list = [[] for col in range(num_col)]

    for col in range(num_col):
        ppcnt = init_ppcnt_list[col]
        for rank in range(ppcnt):
            node = Compressor(stage=-1, column=col, rank=rank, node_type=CompressorType.PI)
            pp = Connection(src_node=node, src_pin=PinType.PI)

            global_node_list.append(node)
            global_pp_list.append(pp)
            pp_list[col].append(pp)

    num_stage = pp_gen.num_min_stage
    pp_array = pp_list_to_pp_array(pp_list)

    # update PP array by compressing 3 rows into 2 rows
    # we maintain a valid pp_list at every iteration
    for stage in range(num_stage):
        pp_array = pp_list_to_pp_array(pp_list)
        num_row_groups = math.floor(len(pp_array) / 3)
        comp_rank_list = [0 for col in range(num_col)]
        new_pp_list = [[] for col in range(num_col)]

        # # debug: print every pp array
        # print(f'Stage {stage}:')
        # for row in range(len(pp_array)):
        #     for col in reversed(range(len(pp_array[0]))):
        #         if pp_array[row][col] is not None:
        #             print("1", end=" ")
        #         else:
        #             print("0", end=" ")
        #     print()

        for rg in range(num_row_groups):
            old_pp_rg_list = pp_array_to_pp_list(pp_array[3*rg:3*rg+3])
            new_pp_rg_list = [[] for col in range(num_col)]

            for col in range(num_col):
                if len(old_pp_rg_list[col]) == 0:
                    continue
                elif len(old_pp_rg_list[col]) == 1:
                    new_pp_rg_list[col].append(old_pp_rg_list[col][0])
                elif len(old_pp_rg_list[col]) == 2:
                    node = Compressor(stage=stage, column=col, rank=comp_rank_list[col], node_type=CompressorType.HA)
                    comp_rank_list[col] += 1
                    global_node_list.append(node)
                    old_pp_rg_list[col][0].set_dst_node(node, pin=PinType.A)
                    old_pp_rg_list[col][1].set_dst_node(node, pin=PinType.B)
                    sum_pp = Connection(src_node=node, src_pin=PinType.S)
                    cout_pp = Connection(src_node=node, src_pin=PinType.CO)
                elif len(old_pp_rg_list[col]) == 3:
                    node = Compressor(stage=stage, column=col, rank=comp_rank_list[col], node_type=CompressorType.FA)
                    comp_rank_list[col] += 1
                    global_node_list.append(node)
                    old_pp_rg_list[col][0].set_dst_node(node, pin=PinType.A)
                    old_pp_rg_list[col][1].set_dst_node(node, pin=PinType.B)
                    old_pp_rg_list[col][2].set_dst_node(node, pin=PinType.CI)
                    sum_pp = Connection(src_node=node, src_pin=PinType.S)
                    cout_pp = Connection(src_node=node, src_pin=PinType.CO)
                else:
                    raise RuntimeError(f'Invalid input PP list length: {len(old_pp_rg_list[col])}')
                
                if len(old_pp_rg_list[col]) in (2, 3):
                    global_pp_list.append(sum_pp)
                    new_pp_rg_list[col].append(sum_pp)
                    if col + 1 < num_col:
                        global_pp_list.append(cout_pp)
                        new_pp_rg_list[col + 1].append(cout_pp)

            for col in range(num_col):
                new_pp_list[col].extend(new_pp_rg_list[col])

        # last rows that get no compression
        if len(pp_array) % 3 != 0:
            remain_pp_list = pp_array_to_pp_list(pp_array[3*num_row_groups:])
            for col in range(num_col):
                new_pp_list[col].extend(remain_pp_list[col])

        # update pp list
        pp_list = new_pp_list

    # assign remaining PPs to output
    pp_array = pp_list_to_pp_array(pp_list)
    assert len(pp_array) == 2
    for col in range(num_col):
        for row in range(2):
            node = Compressor(stage=num_stage, column=col, rank=row, node_type=CompressorType.PO)
            global_node_list.append(node)
            if pp_array[row][col] is not None:
                pp_array[row][col].set_dst_node(node, pin=PinType.PO)
        
    # create a networkx graph
    G = nx.DiGraph()
    for node in global_node_list:
        G.add_node(str(node), instance_name=node.instance_name, **asdict(node))
    for pp in global_pp_list:
        G.add_edge(*pp.edge, **asdict(pp))

    return CompTreeGraphView(G, pp_gen)


def get_dadda_graph_view(pp_gen: PartialProductGenerator) -> CompTreeGraphView:
    """
        Generate a DAG representing dadda multiplier graph
    """

    global_node_list = []
    global_pp_list = []

    # initial pp list
    init_ppcnt_list = pp_gen.get_init_ppcnt_list()
    num_col = pp_gen.num_column
    pp_list = [[] for col in range(num_col)]

    for col in range(num_col):
        ppcnt = init_ppcnt_list[col]
        for rank in range(ppcnt):
            node = Compressor(stage=-1, column=col, rank=rank, node_type=CompressorType.PI)
            pp = Connection(src_node=node, src_pin=PinType.PI)

            global_node_list.append(node)
            global_pp_list.append(pp)
            pp_list[col].append(pp)

    num_stage = pp_gen.num_min_stage

    # update PP list by compressing each column
    # we maintain a valid pp_list at every iteration
    for stage, max_height in enumerate(reversed(get_maximum_height_sequence(num_stage))):
        new_pp_list = [[] for col in range(num_col)]

        for col in range(num_col):
            # compress until max height
            while len(pp_list[col]) + len(new_pp_list[col]) > max_height:
                # apply HA
                if len(pp_list[col]) + len(new_pp_list[col]) == max_height + 1:
                    node = Compressor(stage=stage, column=col, rank=len(new_pp_list[col]), node_type=CompressorType.HA)
                    global_node_list.append(node)
                    
                    pp_list[col][0].set_dst_node(node, pin=PinType.A)
                    pp_list[col][1].set_dst_node(node, pin=PinType.B)
                    pp_list[col].pop(0)
                    pp_list[col].pop(0)

                    sum_pp = Connection(src_node=node, src_pin=PinType.S)
                    cout_pp = Connection(src_node=node, src_pin=PinType.CO)
                    global_pp_list.append(sum_pp)
                    new_pp_list[col].append(sum_pp)
                    if col + 1 < num_col:
                        global_pp_list.append(cout_pp)
                        new_pp_list[col + 1].append(cout_pp)
                # apply FA
                else:
                    node = Compressor(stage=stage, column=col, rank=len(new_pp_list[col]), node_type=CompressorType.FA)
                    global_node_list.append(node)
                    
                    pp_list[col][0].set_dst_node(node, pin=PinType.A)
                    pp_list[col][1].set_dst_node(node, pin=PinType.B)
                    pp_list[col][2].set_dst_node(node, pin=PinType.CI)
                    pp_list[col].pop(0)
                    pp_list[col].pop(0)
                    pp_list[col].pop(0)

                    sum_pp = Connection(src_node=node, src_pin=PinType.S)
                    cout_pp = Connection(src_node=node, src_pin=PinType.CO)
                    global_pp_list.append(sum_pp)
                    new_pp_list[col].append(sum_pp)
                    if col + 1 < num_col:
                        global_pp_list.append(cout_pp)
                        new_pp_list[col + 1].append(cout_pp)

        # update pp list
        for col in range(num_col):
            pp_list[col].extend(new_pp_list[col])

    # assign remaining PPs to output
    pp_array = pp_list_to_pp_array(pp_list)
    assert len(pp_array) == 2
    for col in range(num_col):
        for row in range(2):
            node = Compressor(stage=num_stage, column=col, rank=row, node_type=CompressorType.PO)
            global_node_list.append(node)
            if pp_array[row][col] is not None:
                pp_array[row][col].set_dst_node(node, pin=PinType.PO)
        
    # create a networkx graph
    G = nx.DiGraph()
    for node in global_node_list:
        G.add_node(str(node), instance_name=node.instance_name, **asdict(node))
    for pp in global_pp_list:
        G.add_edge(*pp.edge, **asdict(pp))

    return CompTreeGraphView(G, pp_gen)