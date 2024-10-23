import os
import numpy as np
import networkx as nx
import pickle as pkl
from copy import deepcopy

from .pp_generator import PartialProductGenerator, AndPPGenerator
from .comp_estimator import CompressorEstimator, ArchCompEstimator
from .mult_utils import MultGraphNode, MultGraphEdge, annotate_delay

from utils import mkdir, create_hash

class CompTreeGraphView():

    def __init__(
        self,
        m: int,
        n: int,
        mult_graph: nx.DiGraph,
        pp_gen: PartialProductGenerator = None,
        comp_estimator: CompressorEstimator = None
    ) -> None:
        self.m = m
        self.n = n
        self.mult_graph = mult_graph
        self.pp_gen = pp_gen if pp_gen is not None else AndPPGenerator(m, n)
        self.comp_estimator = comp_estimator if comp_estimator is not None else ArchCompEstimator()

        assert m >= 2
        assert n >= 2
        self._validate(mult_graph)

    def _validate(self, mult_graph: nx.DiGraph) -> bool:
        # the graph should be acyclic
        if not nx.is_directed_acyclic_graph(mult_graph): return False

        # check the connection of every node
        for node in mult_graph.nodes:
            in_nodes = list(mult_graph.predecessors(node))
            out_nodes = list(mult_graph.successors(node))
            in_edges = list(mult_graph.in_edges(node))
            out_edges = list(mult_graph.out_edges(node))

            if node.node_type == MultGraphNode.NODE_TYPE_FA:
                if len(in_nodes) != 3: return False
                if len(out_nodes) != 2: return False

            elif node.node_type == MultGraphNode.NODE_TYPE_HA:
                if len(in_nodes) != 2: return False
                if len(out_nodes) != 2: return False

            elif node.node_type == MultGraphNode.NODE_TYPE_INIT_PP:
                if len(in_nodes) != 0: return False
                if len(out_nodes) != 1: return False

            elif node.node_type == MultGraphNode.NODE_TYPE_OUTPUT:
                if len(in_nodes) != 1: return False
                if len(out_nodes) != 0: return False

            else:
                return False

        # check init pp consistency with pp_gen
        graph_init_pp_list = [0 for _ in range(self.m + self.n)]
        for node in mult_graph.nodes:
            if node.node_type == MultGraphNode.NODE_TYPE_INIT_PP:
                column, rank = node.column, node.rank
                graph_init_pp_list[column] += 1

        ppgen_init_pp_list = self.pp_gen.get_init_ppcnt_list()
        for i, j in zip(graph_init_pp_list, ppgen_init_pp_list):
            if i != j: return False

        return True
    
    @classmethod
    def get_feature_len(cls) -> int:
        """
            The length of feature vector for each node
        """
        return max(cls.get_regression_features().values())[1]
    
    @classmethod
    def get_classification_features(cls) -> dict:
        """
            The features for classification loss in MAE
        """
        feats = dict()
        idx = 0

        feats['invalid'] = (idx, idx + 1)
        idx += 1
        
        feats['node_type'] = (idx, idx + MultGraphNode.NUM_PRED_NODE_TYPE - 1)
        idx += MultGraphNode.NUM_PRED_NODE_TYPE - 1

        feats['stage'] = (idx, idx + MultGraphNode.NUM_MAX_STAGE)
        idx += MultGraphNode.NUM_MAX_STAGE

        feats['pin_A_type'] = (idx, idx + MultGraphEdge.NUM_PRED_PIN_TYPE)
        idx += MultGraphEdge.NUM_PRED_PIN_TYPE

        feats['pin_B_type'] = (idx, idx + MultGraphEdge.NUM_PRED_PIN_TYPE)
        idx += MultGraphEdge.NUM_PRED_PIN_TYPE

        feats['pin_CI_type'] = (idx, idx + MultGraphEdge.NUM_PRED_PIN_TYPE)
        idx += MultGraphEdge.NUM_PRED_PIN_TYPE

        return feats
    
    @classmethod
    def get_regression_features(cls) -> dict:
        feats = dict()
        idx = max(cls.get_classification_features().values())[1]

        feats['delay_A'] = (idx, idx + 1)
        idx += 1

        feats['delay_B'] = (idx, idx + 1)
        idx += 1

        feats['delay_CI'] = (idx, idx + 1)
        idx += 1
        
        return feats
    
    @property
    def max_height(self) -> int:
        """
            The maximum compressor to use in one column
            (I haven't prove this is the correct amount)
        """
        return self.m + self.n

    @property
    def max_delay(self) -> float:
        """
            The maximum delay of the graph
        """
        # TODO: use comp estimator to calculate the delay
        return max([delay for u, v, delay in self.mult_graph.edges.data('delay')])
    
    @property
    def num_stage(self) -> int:
        return max([node.stage for node in self.mult_graph.nodes])

    @property
    def num_full_adder(self) -> int:
        return len([node for node in self.mult_graph.nodes if node.node_type == MultGraphNode.NODE_TYPE_FA])
    
    @property
    def num_half_adder(self) -> int:
        return len([node for node in self.mult_graph.nodes if node.node_type == MultGraphNode.NODE_TYPE_HA])

    @property
    def area(self):
        fa_area = self.comp_estimator.get_fa_area()
        ha_area = self.comp_estimator.get_ha_area()
        return fa_area * self.num_full_adder + ha_area * self.num_half_adder
    
    @property
    def hash(self):
        node_str = ''.join(sorted(map(str, self.mult_graph.nodes)))
        edge_str = ''.join(sorted(map(lambda e: f'{str(e[0])},{str(e[1])}', self.mult_graph.edges)))
        return create_hash(f'{node_str}{edge_str}')
    
    def to_stage_view(self) -> np.ndarray:
        """
            Convert the graph into a stage-like representation
        """
        m, n = self.m, self.n
        stage_array = np.zeros((m + n, 2, self.num_stage), dtype=np.int64)

        for node in self.mult_graph.nodes:
            if node.node_type not in (MultGraphNode.NODE_TYPE_FA, MultGraphNode.NODE_TYPE_HA):
                continue

            type_idx = 0 if node.node_type == MultGraphNode.NODE_TYPE_FA else 1

            stage_array[node.column, type_idx, node.stage] += 1

        return stage_array
    
    def encode_raw_data(self) -> np.ndarray:
        """
            Convert the graph into a image-like representation
        """
        m, n = self.m, self.n
        data = np.zeros((self.get_feature_len(), self.max_height, self.max_height))  # middle is column
        assert m + n == self.max_height

        sorted_nodes = sorted(self.mult_graph.nodes, key=lambda node: (node.stage, node.column, node.rank))
        column_idx_list = [0 for _ in range(m + n)]

        for node in sorted_nodes:
            if node.node_type not in (MultGraphNode.NODE_TYPE_FA, MultGraphNode.NODE_TYPE_HA):
                continue

            # node attr
            feat_node_type = np.eye(MultGraphNode.NUM_PRED_NODE_TYPE)[node.node_type]
            feat_stage     = np.eye(MultGraphNode.NUM_MAX_STAGE)[node.stage]
            feat_column    = np.array([node.column], dtype=np.int64)
            feat_rank      = np.array([node.rank], dtype=np.int64)

            # edge attr
            feat_pin_A  = np.eye(MultGraphEdge.NUM_PRED_PIN_TYPE)[MultGraphEdge.PIN_TYPE_INVALID]
            feat_pin_B  = np.eye(MultGraphEdge.NUM_PRED_PIN_TYPE)[MultGraphEdge.PIN_TYPE_INVALID]
            feat_pin_CI = np.eye(MultGraphEdge.NUM_PRED_PIN_TYPE)[MultGraphEdge.PIN_TYPE_INVALID]
            feat_delay_A = np.zeros((1,), dtype=np.int64)
            feat_delay_B = np.zeros((1,), dtype=np.int64)
            feat_delay_CI = np.zeros((1,), dtype=np.int64)

            for edge in self.mult_graph.in_edges(node):
                edge: MultGraphEdge
                src_pin = np.eye(MultGraphEdge.NUM_PRED_PIN_TYPE)[self.mult_graph.edges[edge]['src_pin']]
                delay   = np.array([self.mult_graph.edges[edge]['delay']], dtype=np.int64)
                if self.mult_graph.edges[edge]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_A:
                    feat_pin_A = src_pin
                    feat_delay_A = delay
                elif self.mult_graph.edges[edge]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_B:
                    feat_pin_B = src_pin
                    feat_delay_B = delay
                elif self.mult_graph.edges[edge]['dst_pin'] == MultGraphEdge.PIN_TYPE_COMP_CI:
                    feat_pin_CI = src_pin
                    feat_delay_CI = delay

            feat = np.concatenate([
                feat_node_type, feat_stage, 
                # feat_column, feat_rank,
                feat_pin_A, feat_pin_B, feat_pin_CI,
                feat_delay_A, feat_delay_B, feat_delay_CI
            ], axis=0)
            feat = np.reshape(feat, (self.get_feature_len(), 1, 1))

            assert feat.shape[0] == self.get_feature_len()
            col_idx = column_idx_list[node.column]
            data[:, node.column:node.column+1, col_idx:col_idx+1] = feat
            column_idx_list[node.column] += 1

        # for the remaining invalid nodes, we need to add a placeholder feature
        for col in range(m + n):
            for col_idx in range(column_idx_list[col], self.max_height):

                # node attr
                feat_node_type = np.eye(MultGraphNode.NUM_PRED_NODE_TYPE)[MultGraphNode.NODE_TYPE_INVALID]
                feat_stage     = np.eye(MultGraphNode.NUM_MAX_STAGE)[-1]
                # feat_column    = np.array([node.column], dtype=np.int64)
                # feat_rank      = np.array([node.rank], dtype=np.int64)

                # edge attr
                feat_pin_A  = np.eye(MultGraphEdge.NUM_PRED_PIN_TYPE)[MultGraphEdge.PIN_TYPE_INVALID]
                feat_pin_B  = np.eye(MultGraphEdge.NUM_PRED_PIN_TYPE)[MultGraphEdge.PIN_TYPE_INVALID]
                feat_pin_CI = np.eye(MultGraphEdge.NUM_PRED_PIN_TYPE)[MultGraphEdge.PIN_TYPE_INVALID]
                feat_delay_A = np.zeros((1,), dtype=np.int64)
                feat_delay_B = np.zeros((1,), dtype=np.int64)
                feat_delay_CI = np.zeros((1,), dtype=np.int64)

                feat = np.concatenate([
                    feat_node_type, feat_stage, 
                    # feat_column, feat_rank,
                    feat_pin_A, feat_pin_B, feat_pin_CI,
                    feat_delay_A, feat_delay_B, feat_delay_CI
                ], axis=0)
                feat = np.reshape(feat, (self.get_feature_len(), 1, 1))

                assert feat.shape[0] == self.get_feature_len()
                data[:, col:col+1, col_idx:col_idx+1] = feat

        return data

    def generate_verilog(self) -> str:
        """
            Generate verilog code for the graph
        """
        m, n = self.m, self.n

        codes = """
module FullAdder(
  input   io_a,
  input   io_b,
  input   io_ci,
  output  io_s,
  output  io_co
);
  wire  a_xor_b = io_a ^ io_b;
  wire  a_and_b = io_a & io_b;
  wire  a_and_cin = io_a & io_ci;
  wire  b_and_cin = io_b & io_ci;
  wire  _T_1 = a_and_b | b_and_cin;
  assign io_s = a_xor_b ^ io_ci;
  assign io_co = _T_1 | a_and_cin;
endmodule

module HalfAdder(
  input   io_a,
  input   io_b,
  output  io_s,
  output  io_co
);
  assign io_s = io_a ^ io_b;
  assign io_co = io_a & io_b;
endmodule
"""

        # io interface
        codes += "module CompressorTree(\n"
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "  input [%d:0] io_pp_%d,\n" % (total_pp_bits-1, col)
        codes += "  output [%d:0] io_augend,\n" % (m+n-1)
        codes += "  output [%d:0] io_addend\n" % (m+n-1)
        codes += ");\n"
        
        for node in self.mult_graph.nodes:
            if node.node_type == MultGraphNode.NODE_TYPE_INIT_PP:
               continue
            elif node.node_type == MultGraphNode.NODE_TYPE_OUTPUT:
                in_edges = list(self.mult_graph.in_edges(node))
                if len(in_edges) == 0:  # we handle len == 1 later
                    codes += f"  assign {node.instance_name} = 1'b0;\n"
            elif node.node_type == MultGraphNode.NODE_TYPE_FA:
                codes += f"""  wire {node.instance_name}_io_a;
  wire {node.instance_name}_io_b;
  wire {node.instance_name}_io_ci;
  wire {node.instance_name}_io_s;
  wire {node.instance_name}_io_co;
  FullAdder {node.instance_name}(
    .io_a({node.instance_name}_io_a),
    .io_b({node.instance_name}_io_b),
    .io_ci({node.instance_name}_io_ci),
    .io_s({node.instance_name}_io_s),
    .io_co({node.instance_name}_io_co)
  );
"""
            elif node.node_type == MultGraphNode.NODE_TYPE_HA:
                codes += f"""  wire {node.instance_name}_io_a;
  wire {node.instance_name}_io_b;
  wire {node.instance_name}_io_s;
  wire {node.instance_name}_io_co;
  HalfAdder {node.instance_name}(
    .io_a({node.instance_name}_io_a),
    .io_b({node.instance_name}_io_b),
    .io_s({node.instance_name}_io_s),
    .io_co({node.instance_name}_io_co)
  );
"""
                
        for edge in self.mult_graph.edges:
            src_node = edge[0]
            dst_node = edge[1]

            src_pin_name_map = {
                # MultGraphEdge.PIN_TYPE_COMP_A:  f"{src_node.instance_name}_io_a",
                # MultGraphEdge.PIN_TYPE_COMP_B:  f"{src_node.instance_name}_io_b",
                # MultGraphEdge.PIN_TYPE_COMP_CI: f"{src_node.instance_name}_io_ci",
                MultGraphEdge.PIN_TYPE_COMP_S:  f"{src_node.instance_name}_io_s",
                MultGraphEdge.PIN_TYPE_COMP_CO: f"{src_node.instance_name}_io_co",
                MultGraphEdge.PIN_TYPE_INIT_PP: src_node.instance_name,
                # MultGraphEdge.PIN_TYPE_OUTPUT:  src_node.instance_name,
            }

            dst_pin_name_map = {
                MultGraphEdge.PIN_TYPE_COMP_A:  f"{dst_node.instance_name}_io_a",
                MultGraphEdge.PIN_TYPE_COMP_B:  f"{dst_node.instance_name}_io_b",
                MultGraphEdge.PIN_TYPE_COMP_CI: f"{dst_node.instance_name}_io_ci",
                # MultGraphEdge.PIN_TYPE_COMP_S:  f"{dst_node.instance_name}_io_s",
                # MultGraphEdge.PIN_TYPE_COMP_CO: f"{dst_node.instance_name}_io_co",
                # MultGraphEdge.PIN_TYPE_INIT_PP: dst_node.instance_name,
                MultGraphEdge.PIN_TYPE_OUTPUT:  dst_node.instance_name,
            }

            src_pin_name = src_pin_name_map[self.mult_graph.edges[edge]['src_pin']]
            dst_pin_name = dst_pin_name_map[self.mult_graph.edges[edge]['dst_pin']]

            codes += "  assign %s = %s;\n" % (dst_pin_name, src_pin_name)

        codes += "endmodule\n"

        return codes

    def dump_data(self, save_dir: str, dump_graph: bool = False):
        """
            Dump the data into the data_dir
        """

        mkdir(save_dir)

        stage_array = self.to_stage_view()
        count_array = stage_array.sum(axis=-1, keepdims=False)

        # dump stage array
        stage_array_path = os.path.join(save_dir, 'stage_array.npy')
        np.save(stage_array_path, stage_array)

        # dump count array
        count_array_path = os.path.join(save_dir, 'count_array.npy')
        np.save(count_array_path, count_array)

        # dump graph
        if dump_graph:
            graph_path = os.path.join(save_dir, 'graph.pkl')
            with open(graph_path, 'wb') as f:
                pkl.dump(self.mult_graph, f)

    def mutate(self, column: int, stage: int, na_rank: int, nb_rank: int, na_pin: int, nb_pin: int) -> nx.DiGraph:
        """
            Swap input connection of two nodes at the same column & stage
        """

        # duplicate a graph
        g = deepcopy(self.mult_graph)

        # find candidate nodes
        nodes = [n for n in g.nodes if n.column == column and n.stage == stage]
        if len(nodes) == 0:
            return None

        # find the nodes
        node_a = None
        node_b = None
        na_rank = na_rank % len(nodes)
        nb_rank = nb_rank % len(nodes)
        for node in nodes:
            if node.column == column and node.stage == stage and node.rank == na_rank:
                node_a = node
            if node.column == column and node.stage == stage and node.rank == nb_rank:
                node_b = node
        if node_a is None or node_b is None:
            return None

        # preprocess selected pin
        fa_pi_list = [MultGraphEdge.PIN_TYPE_COMP_A, MultGraphEdge.PIN_TYPE_COMP_B, MultGraphEdge.PIN_TYPE_COMP_CI]
        ha_pi_list = [MultGraphEdge.PIN_TYPE_COMP_A, MultGraphEdge.PIN_TYPE_COMP_B]

        if node_a.node_type == MultGraphNode.NODE_TYPE_FA:
            if na_pin not in fa_pi_list:
                return None
        elif node_a.node_type == MultGraphNode.NODE_TYPE_HA:
            if na_pin not in ha_pi_list:
                return None
        else:
            return None

        if node_b.node_type == MultGraphNode.NODE_TYPE_FA:
            if nb_pin not in fa_pi_list:
                return None
        elif node_b.node_type == MultGraphNode.NODE_TYPE_HA:
            if nb_pin not in ha_pi_list:
                return None
        else:
            return None

        if node_a == node_b and na_pin == nb_pin:
            return None

        # find predecessor nodes
        pred_a = None
        pred_b = None
        pa_pin = None
        pb_pin = None
        for pred in g.predecessors(node_a):
            if g.edges[pred, node_a]['dst_pin'] == na_pin:
                pred_a = pred
                pa_pin = g.edges[pred, node_a]['src_pin']
        for pred in g.predecessors(node_b):
            if g.edges[pred, node_b]['dst_pin'] == nb_pin:
                pred_b = pred
                pb_pin = g.edges[pred, node_b]['src_pin']
        if pred_a is None or pred_b is None:
            return None
        
        # replace interconnections
        g.remove_edge(pred_a, node_a)
        g.remove_edge(pred_b, node_b)
        g.add_edge(pred_a, node_b)
        g.add_edge(pred_b, node_a)
        g.edges[pred_a, node_b]['src_pin'] = pa_pin
        g.edges[pred_b, node_a]['src_pin'] = pb_pin
        g.edges[pred_a, node_b]['dst_pin'] = nb_pin
        g.edges[pred_b, node_a]['dst_pin'] = na_pin
        if g.nodes[pred_a]['node_type'] == MultGraphNode.NODE_TYPE_INIT_PP:
            g.edges[pred_a, node_b]['delay'] = 0
        if g.nodes[pred_b]['node_type'] == MultGraphNode.NODE_TYPE_INIT_PP:
            g.edges[pred_b, node_a]['delay'] = 0

        return g
    
    def annotate_delay(self):
        """
            Annotate the delay of each edge
        """
        annotate_delay(self.mult_graph, self.comp_estimator)

    def get_critical_path(self) -> list:
        """
            Find the critical path of the graph
        """
        output_nodes = [n for n in self.mult_graph.nodes if n.node_type == MultGraphNode.NODE_TYPE_OUTPUT]

        # find output nodes' predecessors with largest delay
        max_delay = 0
        max_node = None
        max_pred = None

        for node in output_nodes:
            for pred in self.mult_graph.predecessors(node):
                delay = self.mult_graph.edges[pred, node]['delay']
                if delay > max_delay:
                    max_delay = delay
                    max_node  = node
                    max_pred  = pred

        assert max_node is not None
        assert max_pred is not None

        # construct the critical path
        critical_path = []
        
        while max_pred.node_type != MultGraphNode.NODE_TYPE_INIT_PP:
            critical_path.append(max_pred)

            # check critical edge type to find previous node
            edge_type = self.mult_graph.edges[max_pred, max_node]['src_pin']

            if edge_type == MultGraphEdge.PIN_TYPE_COMP_S:
                pred_edge_type = self.mult_graph.nodes[max_pred]['s_critical_transition']
            elif edge_type == MultGraphEdge.PIN_TYPE_COMP_CO:
                pred_edge_type = self.mult_graph.nodes[max_pred]['co_critical_transition']
            else:
                raise ValueError(f'Unknown edge type: {edge_type}')
            
            next_pred = [pred for pred in self.mult_graph.predecessors(max_pred) 
                            if self.mult_graph.edges[pred, max_pred]['dst_pin'] == pred_edge_type][0]
            max_pred, max_node = next_pred, max_pred

        return list(reversed(critical_path))
