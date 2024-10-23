import numpy as np
import networkx as nx
from itertools import product

from .pp_generator import PartialProductGenerator, AndPPGenerator
from .comp_estimator import CompressorEstimator, ArchCompEstimator
from .mult_utils import MultGraphNode, MultGraphEdge, validate_stage_array

from utils import assert_error, info, create_hash

class CompTreeStageView():

    def __init__(self,
                m: int,
                n: int,
                stage_array: np.ndarray,
                pp_gen: PartialProductGenerator = None,
                comp_estimator: CompressorEstimator = None,
                pp_priority: str = 'delay',
    ) -> None:
        self.m = m
        self.n = n
        self.stage_array = stage_array
        self.pp_gen = pp_gen if pp_gen is not None else AndPPGenerator(m, n)
        self.comp_estimator = comp_estimator if comp_estimator is not None else ArchCompEstimator()
        self.pp_priority = pp_priority

        assert m >= 2
        assert n >= 2
        assert self._validate(stage_array)

    # helper properties

    @property
    def num_stage(self):
        return self.stage_array.shape[2]    
    
    @property
    def max_delay(self):
        r = self._to_comp_graph()
        max_delay = max([pp.delay for pp in r['pp_list']])
        return max_delay
    
    @property
    def area(self):
        fa_area = self.comp_estimator.get_fa_area()
        ha_area = self.comp_estimator.get_ha_area()
        return fa_area * np.sum(self.stage_array[:, 0, :]) + ha_area * np.sum(self.stage_array[:, 1, :])
    
    @property
    def hash(self):
        s = str(tuple(np.reshape(self.stage_array, -1)))
        return create_hash(s)

    # helper functions

    def _validate(self, stage_array: np.ndarray) -> bool:
        return validate_stage_array(self.m, self.n, stage_array, self.pp_gen.get_init_ppcnt_list())

    def _sort_node_list(self, node_list: list[MultGraphNode]) -> list[MultGraphNode]:
        return sorted(node_list, key=lambda node: node.rank)
    
    def _sort_pp_list(self, pp_list: list[MultGraphEdge]) -> list[MultGraphEdge]:
        if self.pp_priority == 'delay':
            return sorted(pp_list, key=lambda pp: pp.delay)
        elif self.pp_priority == 'instantiation':
            return sorted(pp_list, key=lambda pp: (pp.src_node.stage, -1 * pp.src_node.column, pp.src_node.rank))

    def _assign_pp_to_node(self, pp_list: list[MultGraphEdge], node_list: list[MultGraphNode]) -> list[list[MultGraphEdge]]:
        """
            Assign available PPs to nodes, and generate new PPs
        """
        m, n = self.m, self.n
        node_list = self._sort_node_list(node_list)
        pp_list = self._sort_pp_list(pp_list)
        new_pp_list = []

        #init a res_col_list to capture res col and sum_pp
        res_col_list=[]
        #init next_col_pp_list for cout
        next_col_pp_list=[]

        idx = 0

        for node in node_list:
            if node.node_type == MultGraphNode.NODE_TYPE_FA:
                pp_list[idx + 0].set_dst_node(node, pin=MultGraphEdge.PIN_TYPE_COMP_A)
                pp_list[idx + 1].set_dst_node(node, pin=MultGraphEdge.PIN_TYPE_COMP_B)
                pp_list[idx + 2].set_dst_node(node, pin=MultGraphEdge.PIN_TYPE_COMP_CI)
                sum_delay, cout_delay = self.comp_estimator.get_fa_delay(pp_list[idx + 0].delay, pp_list[idx + 1].delay, pp_list[idx + 2].delay)
                idx += 3
            elif node.node_type == MultGraphNode.NODE_TYPE_HA:
                pp_list[idx + 0].set_dst_node(node, pin=MultGraphEdge.PIN_TYPE_COMP_A)
                pp_list[idx + 1].set_dst_node(node, pin=MultGraphEdge.PIN_TYPE_COMP_B)
                sum_delay, cout_delay = self.comp_estimator.get_ha_delay(pp_list[idx + 0].delay, pp_list[idx + 1].delay)
                idx += 2
            else:
                raise RuntimeError(f'Invalid node type: {node.node_type}')
            # DEBUG: discard pp whose column >= m + n
            sum_pp = MultGraphEdge(src_node=node, src_pin=MultGraphEdge.PIN_TYPE_COMP_S, delay=sum_delay)
            cout_pp = MultGraphEdge(src_node=node, src_pin=MultGraphEdge.PIN_TYPE_COMP_CO, delay=cout_delay)
            
            new_pp_list.append(sum_pp)
            res_col_list.append(sum_pp)
            if cout_pp.column < m + n :
                new_pp_list.append(cout_pp)
                next_col_pp_list.append(cout_pp)
        
        #get res availble PPs in this column
        if (idx<len(pp_list)):
            res_col_list.extend(pp_list[idx:])
        
        return new_pp_list,res_col_list,next_col_pp_list
    
    def to_count_array(self) -> np.ndarray:
        return np.sum(self.stage_array, axis=2)

    def _to_comp_graph(self) -> dict:
        m, n = self.m, self.n
        stage_array = self.stage_array

        global_node_list = []
        global_pp_list = []

        # initialize PPs and Maintain a global list of available PPs
        init_ppcnt_list = self.pp_gen.get_init_ppcnt_list()
        col_pp_list=[[] for _ in range(m+n)]
        for col in range(m + n):
            for rank in range(init_ppcnt_list[col]):
                node = MultGraphNode(stage=-1, column=col, rank=rank, 
                                     node_type=MultGraphNode.NODE_TYPE_INIT_PP)
                pp = MultGraphEdge(src_node=node, src_pin=MultGraphEdge.PIN_TYPE_INIT_PP,
                                   delay=0)
                global_node_list.append(node)
                global_pp_list.append(pp)
                col_pp_list[col].append(pp)

        # update pp list
        for stage in range(self.num_stage):        
            for col in reversed(range(m + n)):
                num_fa = stage_array[col, 0, stage]
                num_ha = stage_array[col, 1, stage]

                # get available compressors
                cur_node_list = [
                    MultGraphNode(stage=stage, column=col, rank=i, node_type=MultGraphNode.NODE_TYPE_FA)
                    for i in range(num_fa)
                ] + [
                    MultGraphNode(stage=stage, column=col, rank=i, node_type=MultGraphNode.NODE_TYPE_HA)
                    for i in range(num_fa, num_fa + num_ha)
                ]

                # get available PPs
                # TODO: this step may be too slow, rewrite this function
                cur_pp_list = col_pp_list[col]
                # assign PPs to compressor input pins
                new_pp_list,res_col_list,next_col_pp_list = self._assign_pp_to_node(cur_pp_list, cur_node_list)
                # update col_pp_list for next stage reduction
                col_pp_list[col]=res_col_list
                if col<m+n-1:
                    col_pp_list[col+1].extend(next_col_pp_list)


                global_node_list.extend(cur_node_list)
                global_pp_list.extend(new_pp_list)
                

        # assign remaining pp to output
        res_pp_list = [pp for pp in global_pp_list if pp.available]
        res_pp_ranks = [0 for _ in range(m + n)]
        for pp in res_pp_list:
            output_node = MultGraphNode(stage=self.num_stage, column=pp.column, rank=res_pp_ranks[pp.column], \
                                          node_type=MultGraphNode.NODE_TYPE_OUTPUT)
            pp.set_dst_node(output_node, pin=MultGraphEdge.PIN_TYPE_OUTPUT)
            global_node_list.append(output_node)
            res_pp_ranks[pp.column] += 1
            
        assert all([rank <= 2 for rank in res_pp_ranks]), res_pp_ranks

        for col in range(m + n):
            for rank in range(res_pp_ranks[col], 2):
                output_node = MultGraphNode(stage=self.num_stage, column=col, rank=rank, \
                                            node_type=MultGraphNode.NODE_TYPE_OUTPUT)
                global_node_list.append(output_node)

        return {
            'node_list': global_node_list,
            'pp_list': global_pp_list,
        }

    def to_comp_graph(self) -> nx.DiGraph:
        r = self._to_comp_graph()
        global_node_list = r['node_list']
        global_pp_list = r['pp_list']

        # create a networkx graph
        G = nx.DiGraph()
        for node in global_node_list:
            G.add_node(node, **node.attr)
        for pp in global_pp_list:
            G.add_edge(*pp.edge, **pp.attr)

        return G
    
    def mutate(self, column: int, src_stage: int, dst_stage: int) -> np.ndarray:
        """
            Swap FA in src stage and HA in dst stage
            Returns:
                np.ndarry: new config if the mutation is successful,
                None: if the mutation is not successful 
        """

        new_stage_array = self.stage_array.copy()
        if new_stage_array[column, 0, src_stage] == 0: return None
        if new_stage_array[column, 1, dst_stage] == 0: return None
        new_stage_array[column, 0, src_stage] -= 1
        new_stage_array[column, 1, src_stage] += 1
        new_stage_array[column, 0, dst_stage] += 1
        new_stage_array[column, 1, dst_stage] -= 1

        if self._validate(new_stage_array):
            return new_stage_array
        else:
            return None