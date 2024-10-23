import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# helper functions

def get_minimum_stage(n) -> int:
    """
        Get minimum reduction stage with Dadda multiplier's derivation.
        maximize j s.t. d_j < min(m, n) <= d_{j+1}
        https://en.wikipedia.org/wiki/Dadda_multiplier
    """
    assert n <= 256
    d, i = 2, 1
    while True:
        d_next = math.floor(1.5 * d)
        if d < n and d_next >= n:
            return i
        d = d_next
        i += 1

def visualize_stage_array(stage_array):
    """
        Visualize stage array
    """
    n_column, n_comp_type, n_stage = stage_array.shape
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create two subplots side by side

    for c in range(2):
        im = axs[c].imshow(stage_array[:, c, :], cmap='viridis', aspect='auto')
        axs[c].set_xlabel('stage')
        axs[c].set_ylabel('column')
        axs[c].set_title(f'Compressor type {c}')

        # for i in range(n_column):
        #     for j in range(n_stage):
        #         axs[c].text(j, i, stage_array[i, c, j], ha='center', va='center', color='w')

        cbar = fig.colorbar(im, ax=axs[c])
        cbar.set_label('#compressor')

    plt.tight_layout()
    plt.show()

def validate_count_array(m, n, count_array: np.array, init_ppcnt_array: np.array) -> bool:
    if not count_array.shape[0] == m + n: raise RuntimeError(f"Invalid shape: {count_array.shape}")
    if not count_array.shape[1] == 2    : raise RuntimeError(f"Invalid shape: {count_array.shape}")
    if not np.all(count_array >= 0)     : return False

    fa_array, ha_array = count_array[:, 0], count_array[:, 1]
    reduce_array = 2 * fa_array + ha_array
    carry_array = np.concatenate([np.zeros(1, dtype=np.int64), (fa_array + ha_array)[:-1]])

    res_ppcnt_array = init_ppcnt_array - reduce_array + carry_array

    if np.all(np.equal(res_ppcnt_array, 1) + np.equal(res_ppcnt_array, 2)):
        return True
    elif np.all(np.equal(res_ppcnt_array[:-1], 1) + np.equal(res_ppcnt_array[:-1], 2)) and np.all(res_ppcnt_array[-1] == 0):
        # special case for dadda multiplier
        return True
    else:
        print(f"Error at the end, remaining PPs: {res_ppcnt_array}")
        return False

def validate_stage_array(m, n, stage_array: np.array, init_ppcnt_array: np.array) -> bool:
    if not stage_array.shape[0] == m + n: raise RuntimeError(f"Invalid shape: {stage_array.shape}")
    if not stage_array.shape[1] == 2    : raise RuntimeError(f"Invalid shape: {stage_array.shape}")
    if not np.all(stage_array >= 0)     : return False

    flag_mid = True

    ppcnt_array = init_ppcnt_array
    num_stage = stage_array.shape[2]

    for s in range(num_stage):
        fa_array = stage_array[:, 0, s]
        ha_array = stage_array[:, 1, s]

        # remaining PP?
        ppcnt_array -= (3 * fa_array + 2 * ha_array)
        if np.any(ppcnt_array < 0):
            print(f"Error at stage {s}, remaining PPs: {ppcnt_array}")
            flag_mid = False
        
        # generate new pps
        sum_array = fa_array + ha_array
        carry_array = np.concatenate([np.zeros(1, dtype=np.int64), sum_array[:-1]])
        ppcnt_array += (sum_array + carry_array)

    # check remaining PPs
    if np.all(np.equal(ppcnt_array, 1) + np.equal(ppcnt_array, 2)):
        return flag_mid
    elif np.all(np.equal(ppcnt_array[:-1], 1) + np.equal(ppcnt_array[:-1], 2)) and np.all(ppcnt_array[-1] == 0):
        # special case for dadda multiplier
        return flag_mid

    print(f"Error at the end, remaining PPs: {ppcnt_array}")
    return False


# Construct Compressor Tree Graph

class MultGraphNode():
    """
        Node stands for compressor
    """

    # node type constants
    NODE_TYPE_INVALID = 0
    NODE_TYPE_FA      = 1
    NODE_TYPE_HA      = 2
    NODE_TYPE_INIT_PP = 3
    NODE_TYPE_OUTPUT  = 4

    NUM_NODE_TYPE     = 5

    NUM_PRED_NODE_TYPE = 3  # we only predict invalid, FA and HA

    NUM_MAX_STAGE = 16  # we can support multiplier up to 128-bit

    def __init__(
        self,
        stage: int,
        column: int,
        rank: int,
        node_type: int,
    ) -> None:
        self.stage = stage
        self.column = column
        self.rank = rank
        self.node_type = node_type

    def __str__(self) -> str:
        """
            Unique string representation for graph node
        """
        return f"({self.stage}, {self.column}, {self.rank})"
    
    @property
    def instance_name(self) -> str:
        """
            Instance name in verilog codes
        """
        if self.node_type == self.NODE_TYPE_FA:
            return f'FA_{self.stage}_{self.column}_{self.rank}'
        elif self.node_type == self.NODE_TYPE_HA:
            return f'HA_{self.stage}_{self.column}_{self.rank}'
        elif self.node_type == self.NODE_TYPE_INIT_PP:
            return f'io_pp_{self.column}[{self.rank}]'
        elif self.node_type == self.NODE_TYPE_OUTPUT:
            if self.rank == 0:
                return f'io_augend[{self.column}]'
            elif self.rank == 1:
                return f'io_addend[{self.column}]'
            else:
                raise RuntimeError(f'Invalid output rank: {self.rank}')
        else:
            raise RuntimeError(f'Invalid node type: {self.node_type}')
    
    @property
    def attr(self):
        return {
            'stage': self.stage,
            'column': self.column,
            'rank': self.rank,
            'node_type': self.node_type,
            'instance_name': self.instance_name,
        }
    
class MultGraphEdge():
    """
        Edge stands for partial product
    """

    # pin type constants
    PIN_TYPE_INVALID = 0
    PIN_TYPE_INIT_PP = 1
    PIN_TYPE_COMP_S  = 2
    PIN_TYPE_COMP_CO = 3
    PIN_TYPE_COMP_A  = 4
    PIN_TYPE_COMP_B  = 5
    PIN_TYPE_COMP_CI = 6
    PIN_TYPE_OUTPUT  = 7

    NUM_PIN_TYPE     = 8

    NUM_PRED_PIN_TYPE = 4  # we only predict pp, s and co

    def __init__(
        self, 
        src_node: MultGraphNode = None,
        dst_node: MultGraphNode = None,
        src_pin: int = None,
        dst_pin: int = None,
        delay: float = None
    ) -> None:
        self.src_node = src_node
        self.dst_node = dst_node
        self.src_pin = src_pin
        self.dst_pin = dst_pin
        self.delay = delay

    def set_src_node(self, node: MultGraphNode, pin: int) -> None:
        assert self.src_node is None
        self.src_node = node
        self.src_pin = pin

    def set_dst_node(self, node: MultGraphNode, pin: int) -> None:
        assert self.dst_node is None
        self.dst_node = node
        self.dst_pin = pin

    @property
    def column(self):
        assert self.src_node is not None
        assert self.src_pin is not None
        if self.src_pin == self.PIN_TYPE_COMP_S:
            return self.src_node.column
        elif self.src_pin == self.PIN_TYPE_INIT_PP:
            return self.src_node.column
        elif self.src_pin == self.PIN_TYPE_COMP_CO:
            return self.src_node.column + 1
        else:
            raise RuntimeError(f'Invalid src pin type: {self.src_pin}')
    
    @property
    def stage(self):
        """
            From (and including) this stage, this PP is available
        """
        assert self.src_node is not None
        return self.src_node.stage + 1
    
    @property
    def available(self) -> bool:
        return self.dst_node is None

    @property
    def edge(self):
        return (self.src_node, self.dst_node)
    
    @property
    def attr(self):
        return {
            'src_pin': self.src_pin,
            'dst_pin': self.dst_pin,
            'delay': self.delay,
        }
    
def annotate_delay(comp_graph: nx.DiGraph, comp_est) -> None:
    """
        Annotate delay to each node in compressor graph
        TODO: Nangate45 should support this method
    """
    for node in nx.topological_sort(comp_graph):
        if node.node_type == MultGraphNode.NODE_TYPE_FA:
            input_delays = [None, None, None]
            for e in comp_graph.in_edges(node):
                if comp_graph.edges[e]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_A:
                    input_delays[0] = comp_graph.edges[e]['delay']
                elif comp_graph.edges[e]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_B:
                    input_delays[1] = comp_graph.edges[e]['delay']
                elif comp_graph.edges[e]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_CI:
                    input_delays[2] = comp_graph.edges[e]['delay']
                else:
                    raise RuntimeError(f'Invalid dst pin type: {comp_graph.edges[e]["dst_pin"]}')
            s_delay, co_delay = comp_est.get_fa_delay(*input_delays)
            for e in comp_graph.out_edges(node):
                if comp_graph.edges[e]['src_pin'] == MultGraphEdge.PIN_TYPE_COMP_S:
                    comp_graph.edges[e]['delay'] = s_delay
                elif comp_graph.edges[e]['src_pin'] == MultGraphEdge.PIN_TYPE_COMP_CO:
                    comp_graph.edges[e]['delay'] = co_delay
                else:
                    raise RuntimeError(f'Invalid dst pin type: {comp_graph.edges[e]["dst_pin"]}')
            # mark critical transition
            s_trans, co_trans = comp_est.get_fa_critical_transition(*input_delays)
            comp_graph.nodes[node]['s_critical_transition'] = s_trans
            comp_graph.nodes[node]['co_critical_transition'] = co_trans
        elif node.node_type == MultGraphNode.NODE_TYPE_HA:
            input_delays = [None, None]
            for e in comp_graph.in_edges(node):
                if comp_graph.edges[e]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_A:
                    input_delays[0] = comp_graph.edges[e]['delay']
                elif comp_graph.edges[e]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_B:
                    input_delays[1] = comp_graph.edges[e]['delay']
                else:
                    raise RuntimeError(f'Invalid dst pin type: {comp_graph.edges[e]["dst_pin"]}')
            s_delay, co_delay = comp_est.get_ha_delay(*input_delays)
            for e in comp_graph.out_edges(node):
                if comp_graph.edges[e]['src_pin'] == MultGraphEdge.PIN_TYPE_COMP_S:
                    comp_graph.edges[e]['delay'] = s_delay
                elif comp_graph.edges[e]['src_pin'] == MultGraphEdge.PIN_TYPE_COMP_CO:
                    comp_graph.edges[e]['delay'] = co_delay
                else:
                    raise RuntimeError(f'Invalid dst pin type: {comp_graph.edges[e]["dst_pin"]}')
            # mark critical transition
            s_trans, co_trans = comp_est.get_ha_critical_transition(*input_delays)
            comp_graph.nodes[node]['s_critical_transition'] = s_trans
            comp_graph.nodes[node]['co_critical_transition'] = co_trans
        elif node.node_type == MultGraphNode.NODE_TYPE_INIT_PP:
            continue
        elif node.node_type == MultGraphNode.NODE_TYPE_OUTPUT:
            continue
        else:
            raise RuntimeError(f'Invalid node type: {node.node_type}')
    
# Exceptions 

class InvalidActionError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
    
class LegalizationError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
