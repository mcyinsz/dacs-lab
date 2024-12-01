import abc
import math
import numpy as np
import random
import networkx as nx
from typing import List
from itertools import permutations
from dataclasses import asdict

from .stage_view import CompTreeStageView
from .graph_view import CompTreeGraphView
from .graph_utils import Compressor, CompressorType, PinType, Connection, CompressorTimer

class CompTreeStageViewMapper(abc.ABC):
    """
        Abstract class for mapping compressor tree stage view to graph view
    """

    @abc.abstractmethod
    def _assign_pp_to_node(self, compressor_list: List[Compressor], connection_list: List[Connection]) -> None:
        raise NotImplementedError

    def assign_pp_to_node(self, column: int, stage: int) -> None:
        """
            assign the input pins of compressors in a specific slice.
        """
        compressor_list = self.get_compressor_list(column, stage)
        init_pp_list = [conn for conn in self.get_connection_list(column, stage-1)
                            if conn.src_pin == PinType.PI]
        sum_pp_list = [conn for conn in self.get_connection_list(column, stage-1)
                            if conn.src_pin == PinType.S]
        cout_pp_list = [conn for conn in self.get_connection_list(column-1, stage-1)
                            if conn.src_pin == PinType.CO]
        connection_list = init_pp_list + sum_pp_list + cout_pp_list

        assert 3 * len([node for node in compressor_list if node.node_type == CompressorType.FA]) \
             + 2 * len([node for node in compressor_list if node.node_type == CompressorType.HA]) \
             + len([node for node in compressor_list if node.node_type == CompressorType.FT]) \
             + len([node for node in compressor_list if node.node_type == CompressorType.PO]) \
             == len(connection_list), f'Number of compressors and PPs do not match at column {column} stage {stage}'
        
        self._assign_pp_to_node(compressor_list, connection_list)

    
    def __call__(self, stage_view: CompTreeStageView) -> CompTreeGraphView:

        # initialize node and partailly-connected edges
        self.init_slice_dict(stage_view)

        # for each slice, assign PP to nodes
        for stage in range(stage_view.num_stage + 1):
            for column in range(stage_view.num_column):
                # print(f'Assigning PPs to nodes at column {column} stage {stage}')
                self.assign_pp_to_node(column, stage)

        # collect all nodes and edges
        global_node_list = [node for node_list in self.compressor_slice_dict.values() for node in node_list]
        global_edge_list = [pp for pp_list in self.connection_slice_dict.values() for pp in pp_list]

        for conn in global_edge_list:
            assert conn.dst_node is not None, f'PP {conn} is not connected to any node'

        # create a networkx graph
        G = nx.DiGraph()
        for node in global_node_list:
            G.add_node(str(node), instance_name=node.instance_name, **asdict(node))
        for pp in global_edge_list:
            G.add_edge(*pp.edge, **asdict(pp))

        return CompTreeGraphView(G, stage_view.pp_gen)
    
    def init_slice_dict(self, stage_view: CompTreeStageView) -> None:
        """
            create a dictionary to store compressor slices: (column, stage) -> [compressors]
        """
        self.compressor_slice_dict = dict()
        self.connection_slice_dict = dict()

        # PIs
        init_ppcnt_list = stage_view.pp_gen.get_init_ppcnt_list()
        for col in range(stage_view.num_column):
            for rank in range(init_ppcnt_list[col]):
                node = Compressor(stage=-1, column=col, rank=rank, node_type=CompressorType.PI)
                self.get_compressor_list(column=col, stage=-1).append(node)

                edge = Connection(src_node=node, src_pin=PinType.PI)
                self.get_connection_list(column=col, stage=-1).append(edge)                

        # FAs, HAs and FTs
        res_ppcnt_array = stage_view.get_res_ppcnt_array()
        for col in range(stage_view.num_column):
            for stage in range(stage_view.num_stage):
                num_fa = stage_view.stage_array[col, 0, stage]
                num_ha = stage_view.stage_array[col, 1, stage]
                num_ft = res_ppcnt_array[col, stage]

                cur_node_list = [
                    Compressor(stage=stage, column=col, rank=i, node_type=CompressorType.FA)
                    for i in range(num_fa)
                ] + [
                    Compressor(stage=stage, column=col, rank=i, node_type=CompressorType.HA)
                    for i in range(num_fa, num_fa + num_ha)
                ] + [
                    Compressor(stage=stage, column=col, rank=i, node_type=CompressorType.FT)
                    for i in range(num_fa + num_ha, num_fa + num_ha + num_ft)
                ]
                self.get_compressor_list(column=col, stage=stage).extend(cur_node_list)

                cur_edge_list = [
                    Connection(src_node=node, src_pin=PinType.S)
                    for node in cur_node_list
                ] + [
                    Connection(src_node=node, src_pin=PinType.CO)
                    for node in cur_node_list if node.node_type != CompressorType.FT and node.column != stage_view.num_column - 1
                ]
                self.get_connection_list(column=col, stage=stage).extend(cur_edge_list)

        # POs
        final_ppcnt_list = stage_view.to_count_view().get_res_ppcnt_list()
        for col in range(stage_view.num_column):
            for rank in range(final_ppcnt_list[col]):
                node = Compressor(stage=stage_view.num_stage, column=col, rank=rank, node_type=CompressorType.PO)
                self.get_compressor_list(column=col, stage=stage_view.num_stage).append(node)

        
    def get_compressor_list(self, column: int, stage: int) -> List[Compressor]:
        """
            Get compressor list in a specific slice
        """
        k = f'c{column}_s{stage}'
        if k not in self.compressor_slice_dict:
            self.compressor_slice_dict[k] = []
        return self.compressor_slice_dict.get(k)

    def get_connection_list(self, column: int, stage: int) -> List[Connection]:
        """
            Get pp list in a specific slice
        """
        k = f'c{column}_s{stage}'
        if k not in self.connection_slice_dict:
            self.connection_slice_dict[k] = []
        return self.connection_slice_dict.get(k)
    

class SortedCompTreeStageViewMapper(CompTreeStageViewMapper):
    """
        Mapper for serializing compressor tree stage view to graph view
        Concretely, it assigns PPs and compressors in a specific order,
        such as predicted timing (ArithmeticTree), instantiation order (EasyMAC & GOMIL), etc.
    """

    def __init__(self, compressor_timer: CompressorTimer = None, pp_priority: str = 'delay') -> None:
        self.compressor_timer = compressor_timer if compressor_timer is not None else CompressorTimer()
        self.pp_priority = pp_priority

    def _sort_node_list(self, node_list: list[Compressor]) -> list[Compressor]:
        """
            Sort the compressor in a given slice.
            We choose a trivial order: rank, and optimize pp assignment
        """
        return sorted(node_list, key=lambda node: node.rank)
    
    def _sort_pp_list(self, pp_list: list[Connection]) -> list[Connection]:
        """
            Sort the connection list in a specific order.
        """
        if self.pp_priority == 'delay':
            return sorted(pp_list, key=lambda pp: self._get_src_delay(pp))
        elif self.pp_priority == 'instantiation':
            return sorted(pp_list, key=lambda pp: (pp.src_node.stage, -1 * pp.src_node.column, pp.src_node.rank))
        
    def _get_src_delay(self, pp: Connection) -> float:
        """
            Read delay from src node
        """
        node = pp.src_node
        if pp.src_pin == PinType.PI:
            return 0
        elif pp.src_pin == PinType.S:
            return node.sum_delay
        elif pp.src_pin == PinType.CO:
            return node.cout_delay
        else:
            raise RuntimeError(f'Invalid pin type: {pp.src_pin}')
        
    def _assign_pp_to_node_no_sort(self, compressor_list: List[Compressor], connection_list: List[Connection]) -> None:
        """
            Assign available PPs to nodes, and generate new PPs, but does not sort the input lists
        """

        node_list = compressor_list
        pp_list = connection_list

        idx = 0

        for node in node_list:
            if node.node_type == CompressorType.FA:
                pp_list[idx + 0].set_dst_node(node, pin=PinType.A)
                pp_list[idx + 1].set_dst_node(node, pin=PinType.B)
                pp_list[idx + 2].set_dst_node(node, pin=PinType.CI)
                sum_delay, cout_delay = self.compressor_timer.get_fa_delay(
                    a=self._get_src_delay(pp_list[idx + 0]), 
                    b=self._get_src_delay(pp_list[idx + 1]), 
                    ci=self._get_src_delay(pp_list[idx + 2])
                )
                idx += 3
            elif node.node_type == CompressorType.HA:
                pp_list[idx + 0].set_dst_node(node, pin=PinType.A)
                pp_list[idx + 1].set_dst_node(node, pin=PinType.B)
                sum_delay, cout_delay = self.compressor_timer.get_ha_delay(
                    a=self._get_src_delay(pp_list[idx + 0]), 
                    b=self._get_src_delay(pp_list[idx + 1])
                )
                idx += 2
            elif node.node_type == CompressorType.FT:
                pp_list[idx + 0].set_dst_node(node, pin=PinType.A)
                sum_delay, cout_delay = self._get_src_delay(pp_list[idx]), 0
                idx += 1
            elif node.node_type == CompressorType.PO:
                pp_list[idx + 0].set_dst_node(node, pin=PinType.PO)
                sum_delay, cout_delay = self._get_src_delay(pp_list[idx]), 0
                idx += 1
            else:
                raise RuntimeError(f'Invalid node type: {node.node_type}')
            
            node.sum_delay = sum_delay
            node.cout_delay = cout_delay


    def _assign_pp_to_node(self, compressor_list: List[Compressor], connection_list: List[Connection]) -> None:
        """
            Assign available PPs to nodes, and generate new PPs
        """
        node_list = self._sort_node_list(compressor_list)
        pp_list = self._sort_pp_list(connection_list)

        return self._assign_pp_to_node_no_sort(node_list, pp_list)


class DpCompTreeStageViewMapper(SortedCompTreeStageViewMapper):
    """
        Formulate the process as Dynamic Programming (technically not abide by the definition)
        For each slice, we find the optimum assignment that minimize overall delay
    """

    def __init__(self, compressor_timer: CompressorTimer = None, pp_priority: str = 'delay', max_sample_per_step: int = 5000) -> None:
        super().__init__(compressor_timer, pp_priority)
        self.max_sample_per_step = max_sample_per_step


    def _assign_pp_to_node(self, compressor_list: List[Compressor], connection_list: List[Connection]) -> None:
        """
            Assign available PPs to nodes that minimize the overall delay observed by current slice
        """
        node_list = self._sort_node_list(compressor_list)
        connection_list = self._sort_pp_list(connection_list)

        def overall_delay_func() -> float:
            return max([node.sum_delay for node in node_list] + [node.cout_delay for node in node_list] + [0])
        
        # assign initial permutation
        self._assign_pp_to_node_no_sort(node_list, connection_list)
        best_permutation = [conn for conn in connection_list]
        best_overall_delay = overall_delay_func()

        def random_perm_gen(seq):
            for i in range(self.max_sample_per_step):
                yield random.sample(seq, len(seq))

        perm_gen = permutations(connection_list) if math.factorial(len(connection_list)) <= self.max_sample_per_step \
                   else random_perm_gen(connection_list)
        
        for permutation in perm_gen:
            self._assign_pp_to_node_no_sort(node_list, permutation)
            overall_delay = overall_delay_func()

            if overall_delay < best_overall_delay:
                best_overall_delay = overall_delay
                best_permutation = permutation

        # assign the best permutation
        self._assign_pp_to_node_no_sort(node_list, best_permutation)


class LegacyCompTreeStageViewMapper(CompTreeStageViewMapper):
    """
        Legacy mapper for assigning PPs to nodes in a specific order, modified from CompTreeStageView
        This mapper does not insert any FT nodes
    """
    def __init__(self, compressor_timer: CompressorTimer = None, pp_priority: str = 'delay') -> None:
        self.num_column = None
        self.num_stage = None
        self.pp_gen = None
        self.stage_array = None
        self.comp_estimator = compressor_timer if compressor_timer is not None else CompressorTimer()
        self.pp_priority = pp_priority

    def _sort_node_list(self, node_list: list[Compressor]) -> list[Compressor]:
        return sorted(node_list, key=lambda node: node.rank)
    
    def _sort_pp_list(self, pp_list: list[Connection]) -> list[Connection]:
        if self.pp_priority == 'delay':
            return sorted(pp_list, key=lambda pp: pp.delay)
        elif self.pp_priority == 'instantiation':
            return sorted(pp_list, key=lambda pp: (pp.src_node.stage, -1 * pp.src_node.column, pp.src_node.rank))

    def _assign_pp_to_node(self, compressor_list: List[Compressor], connection_list: List[Connection]) -> list[list[Connection]]:
        """
            Assign available PPs to nodes, and generate new PPs
        """
        node_list = self._sort_node_list(compressor_list)
        pp_list = self._sort_pp_list(connection_list)
        new_pp_list = []

        #init a res_col_list to capture res col and sum_pp
        res_col_list=[]
        #init next_col_pp_list for cout
        next_col_pp_list=[]

        idx = 0

        for node in node_list:
            if node.node_type == CompressorType.FA:
                pp_list[idx + 0].set_dst_node(node, pin=PinType.A)
                pp_list[idx + 1].set_dst_node(node, pin=PinType.B)
                pp_list[idx + 2].set_dst_node(node, pin=PinType.CI)
                sum_delay, cout_delay = self.comp_estimator.get_fa_delay(pp_list[idx + 0].delay, pp_list[idx + 1].delay, pp_list[idx + 2].delay)
                idx += 3
            elif node.node_type == CompressorType.HA:
                pp_list[idx + 0].set_dst_node(node, pin=PinType.A)
                pp_list[idx + 1].set_dst_node(node, pin=PinType.B)
                sum_delay, cout_delay = self.comp_estimator.get_ha_delay(pp_list[idx + 0].delay, pp_list[idx + 1].delay)
                idx += 2
            else:
                raise RuntimeError(f'Invalid node type: {node.node_type}')
            # DEBUG: discard pp whose column >= self.num_column
            sum_pp = Connection(src_node=node, src_pin=PinType.S, delay=sum_delay)
            cout_pp = Connection(src_node=node, src_pin=PinType.CO, delay=cout_delay)
            
            new_pp_list.append(sum_pp)
            res_col_list.append(sum_pp)
            if cout_pp.column < self.num_column :
                new_pp_list.append(cout_pp)
                next_col_pp_list.append(cout_pp)
        
        #get res availble PPs in this column
        if (idx<len(pp_list)):
            res_col_list.extend(pp_list[idx:])
        
        return new_pp_list,res_col_list,next_col_pp_list


    def _to_comp_graph(self) -> dict:
        stage_array = self.stage_array

        global_node_list = []
        global_pp_list = []

        # initialize PPs and Maintain a global list of available PPs
        init_ppcnt_list = self.pp_gen.get_init_ppcnt_list()
        col_pp_list=[[] for _ in range(self.num_column)]
        for col in range(self.num_column):
            for rank in range(init_ppcnt_list[col]):
                node = Compressor(stage=-1, column=col, rank=rank, node_type=CompressorType.PI)
                pp = Connection(src_node=node, src_pin=PinType.PI, delay=0)
                global_node_list.append(node)
                global_pp_list.append(pp)
                col_pp_list[col].append(pp)

        # update pp list
        for stage in range(self.num_stage):        
            for col in reversed(range(self.num_column)):
                num_fa = stage_array[col, 0, stage]
                num_ha = stage_array[col, 1, stage]

                # get available compressors
                cur_node_list = [
                    Compressor(stage=stage, column=col, rank=i, node_type=CompressorType.FA)
                    for i in range(num_fa)
                ] + [
                    Compressor(stage=stage, column=col, rank=i, node_type=CompressorType.HA)
                    for i in range(num_fa, num_fa + num_ha)
                ]

                # get available PPs
                # TODO: this step may be too slow, rewrite this function
                cur_pp_list = col_pp_list[col]
                # assign PPs to compressor input pins
                new_pp_list,res_col_list,next_col_pp_list = self._assign_pp_to_node(cur_node_list, cur_pp_list)
                # update col_pp_list for next stage reduction
                col_pp_list[col]=res_col_list
                if col<self.num_column-1:
                    col_pp_list[col+1].extend(next_col_pp_list)


                global_node_list.extend(cur_node_list)
                global_pp_list.extend(new_pp_list)
                

        # assign remaining pp to output
        res_pp_list = [pp for pp in global_pp_list if pp.dst_node is None]
        res_pp_ranks = [0 for _ in range(self.num_column)]
        for pp in res_pp_list:
            output_node = Compressor(stage=self.num_stage, column=pp.column, rank=res_pp_ranks[pp.column], \
                                          node_type=CompressorType.PO)
            pp.set_dst_node(output_node, pin=PinType.PO)
            global_node_list.append(output_node)
            res_pp_ranks[pp.column] += 1
            
        assert all([rank <= 2 for rank in res_pp_ranks]), res_pp_ranks

        for col in range(self.num_column):
            for rank in range(res_pp_ranks[col], 2):
                output_node = Compressor(stage=self.num_stage, column=col, rank=rank, \
                                            node_type=CompressorType.PO)
                global_node_list.append(output_node)

        return {
            'node_list': global_node_list,
            'pp_list': global_pp_list,
        }


    def __call__(self, stage_view: CompTreeStageView) -> CompTreeGraphView:
        self.num_column = stage_view.num_column
        self.num_stage = stage_view.num_stage
        self.stage_array = stage_view.stage_array
        self.pp_gen = stage_view.pp_gen

        r = self._to_comp_graph()
        global_node_list = r['node_list']
        global_edge_list = r['pp_list']

        for conn in global_edge_list:
            assert conn.dst_node is not None, f'PP {conn} is not connected to any node'

        # create a networkx graph
        G = nx.DiGraph()
        for node in global_node_list:
            G.add_node(str(node), instance_name=node.instance_name, **asdict(node))
        for pp in global_edge_list:
            G.add_edge(*pp.edge, **asdict(pp))

        return CompTreeGraphView(G, stage_view.pp_gen)