import numpy as np
import networkx as nx
import os
import pickle as pkl
from copy import deepcopy

from .stage_view import CompTreeStageView
from .graph_utils import Compressor, CompressorType, PinType, Connection
from design.multiplier.pp_generator import PartialProductGenerator
from design.multiplier.mult_utils import InvalidStageViewError, InvalidGraphViewError

from utils import mkdir, create_hash, dump_json

class CompTreeGraphView():

    def __init__(
        self,
        comp_graph: nx.DiGraph,
        pp_gen: PartialProductGenerator,
    ) -> None:
        self.comp_graph = comp_graph
        self.pp_gen = pp_gen

        if not self.validate(comp_graph):
            raise InvalidGraphViewError(f'Invalid graph: {comp_graph}')

    def validate(self, comp_graph: nx.DiGraph) -> bool:
        # the graph should be acyclic
        if not nx.is_directed_acyclic_graph(comp_graph): return False

        # check the connection of every node
        for node in comp_graph.nodes:

            # check IO of each node
            in_nodes = list(comp_graph.predecessors(node))
            out_nodes = list(comp_graph.successors(node))
            node_type = comp_graph.nodes[node]['node_type']
            node_stage = comp_graph.nodes[node]['stage']
            node_column = comp_graph.nodes[node]['column']

            if node_type == CompressorType.FA:
                if len(in_nodes) != 3: return False
                if len(out_nodes) != 2:
                    if not (len(out_nodes) == 1 and node_column == self.num_column - 1):
                        return False
            elif node_type == CompressorType.HA:
                if len(in_nodes) != 2: return False
                if len(out_nodes) != 2:
                    if not (len(out_nodes) == 1 and node_column == self.num_column - 1):
                        return False
            elif node_type == CompressorType.FT:
                if len(in_nodes) != 1: return False
                if len(out_nodes) != 1: return False
            elif node_type == CompressorType.PI:
                if len(in_nodes) != 0: return False
                if len(out_nodes) != 1: return False
            elif node_type == CompressorType.PO:
                if len(in_nodes) != 1 and len(in_nodes) != 0: return False
                if len(out_nodes) != 0: return False
            else:
                return False
            
        # check init pp consistency with pp_gen
        graph_init_pp_list = [0 for _ in range(self.num_column)]
        for node in comp_graph.nodes:
            node_type = comp_graph.nodes[node]['node_type']
            if node_type == CompressorType.PI:
                column = comp_graph.nodes[node]['column']
                graph_init_pp_list[column] += 1

        ppgen_init_pp_list = self.pp_gen.get_init_ppcnt_list()
        for i, j in zip(graph_init_pp_list, ppgen_init_pp_list):
            if i != j: return False

        # check the conversion to stage view
        try:
            stage_array = np.zeros((self.num_column, 2, self.num_stage), dtype=np.int64)
            for node in self.comp_graph.nodes:
                node_type = self.comp_graph.nodes[node]['node_type']
                column = self.comp_graph.nodes[node]['column']
                stage = self.comp_graph.nodes[node]['stage']
                if node_type == CompressorType.FA:
                    type_idx = 0
                elif node_type == CompressorType.HA:
                    type_idx = 1
                else:
                    continue
                stage_array[column, type_idx, stage] += 1
            stage_view = CompTreeStageView(stage_array, self.pp_gen)
        except InvalidStageViewError:
            import traceback
            print(traceback.format_exc())
            return False

        return True
    
    @property
    def num_column(self) -> int:
        return self.pp_gen.num_column

    @property
    def num_stage(self) -> int:
        return max([self.comp_graph.nodes[node]['stage'] for node in self.comp_graph.nodes])

    @property
    def num_full_adder(self) -> int:
        return len([node for node in self.comp_graph.nodes if self.comp_graph.nodes[node]['node_type'] == CompressorType.FA])
    
    @property
    def num_half_adder(self) -> int:
        return len([node for node in self.comp_graph.nodes if self.comp_graph.nodes[node]['node_type'] == CompressorType.HA])
    
    @property
    def hash(self):
        node_str = ''.join(sorted(map(str, self.comp_graph.nodes)))
        edge_str = ''.join(sorted(map(lambda e: f'{str(e[0])},{str(e[1])}', self.comp_graph.edges)))
        return create_hash(f'{node_str}{edge_str}')
    
    def to_stage_view(self) -> CompTreeStageView:
        """
            Convert the graph into a stage-like representation
        """

        stage_array = np.zeros((self.num_column, 2, self.num_stage), dtype=np.int64)

        for node in self.comp_graph.nodes:
            column    = self.comp_graph.nodes[node]['column']
            stage     = self.comp_graph.nodes[node]['stage']
            node_type = self.comp_graph.nodes[node]['node_type']
            if node_type not in (CompressorType.FA, CompressorType.HA):
                continue
            type_idx = 0 if node_type == CompressorType.FA else 1
            stage_array[column, type_idx, stage] += 1

        return CompTreeStageView(stage_array, self.pp_gen)
    
    def generate_verilog(
        self,
        module_name: str = 'CompressorTree',
    ) -> str:
        """
            Generate verilog code for the graph
        """

        fa_name = 'FullAdder'
        ha_name = 'HalfAdder'
        ft_name = 'FallThrough'
        
        # submodules
        codes = "\n".join([
            self.generate_full_adder_verilog(name=fa_name),
            self.generate_half_adder_verilog(name=ha_name),
            self.generate_fall_through_verilog(name=ft_name),
        ])

        # io interface
        codes += "module %s(\n" % module_name
        codes += "  input clock,\n"
        codes += "  input reset,\n"
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "  input [%d:0] io_pp_%d,\n" % (total_pp_bits-1, col)
        codes += "  output [%d:0] io_augend,\n" % (self.num_column-1)
        codes += "  output [%d:0] io_addend\n" % (self.num_column-1)
        codes += ");\n"
        
        # instantiate compressors
        for node in self.comp_graph.nodes:
            node_type = self.comp_graph.nodes[node]['node_type']
            instance_name = self.comp_graph.nodes[node]['instance_name']
                                            
            if node_type == CompressorType.PI:
               continue
            elif node_type == CompressorType.PO:
                continue
            elif node_type == CompressorType.FA:
                codes += f"""  wire {instance_name}_io_a;
  wire {instance_name}_io_b;
  wire {instance_name}_io_ci;
  wire {instance_name}_io_s;
  wire {instance_name}_io_co;
  {fa_name} {instance_name}(
    .io_a({instance_name}_io_a),
    .io_b({instance_name}_io_b),
    .io_ci({instance_name}_io_ci),
    .io_s({instance_name}_io_s),
    .io_co({instance_name}_io_co)
  );
"""
            elif node_type == CompressorType.HA:
                codes += f"""  wire {instance_name}_io_a;
  wire {instance_name}_io_b;
  wire {instance_name}_io_s;
  wire {instance_name}_io_co;
  {ha_name} {instance_name}(
    .io_a({instance_name}_io_a),
    .io_b({instance_name}_io_b),
    .io_s({instance_name}_io_s),
    .io_co({instance_name}_io_co)
  );
"""
            elif node_type == CompressorType.FT:
                codes += f"""  wire {instance_name}_io_a;
  wire {instance_name}_io_s;
  {ft_name} {instance_name}(
    .io_a({instance_name}_io_a),
    .io_s({instance_name}_io_s)
  );
"""
        
        po_status = {f'io_augend[{i}]': False for i in range(self.num_column)}
        po_status.update({f'io_addend[{i}]': False for i in range(self.num_column)})

        for edge in self.comp_graph.edges:

            src_inst_name = self.comp_graph.nodes[edge[0]]['instance_name']
            dst_inst_name = self.comp_graph.nodes[edge[1]]['instance_name']

            src_pin_name_map = {
                # PinType.A:  f"{instance_name}_io_a",
                # PinType.B:  f"{instance_name}_io_b",
                # PinType.CI: f"{instance_name}_io_ci",
                PinType.S:  f"{src_inst_name}_io_s",
                PinType.CO: f"{src_inst_name}_io_co",
                PinType.PI: src_inst_name,
                # PinType.PO:  instance_name,
            }

            dst_pin_name_map = {
                PinType.A:  f"{dst_inst_name}_io_a",
                PinType.B:  f"{dst_inst_name}_io_b",
                PinType.CI: f"{dst_inst_name}_io_ci",
                # PinType.S:  f"{instance_name}_io_s",
                # PinType.CO: f"{instance_name}_io_co",
                # PinType.PI: instance_name,
                PinType.PO:  dst_inst_name,
            }

            src_pin_name = src_pin_name_map[self.comp_graph.edges[edge]['src_pin']]
            dst_pin_name = dst_pin_name_map[self.comp_graph.edges[edge]['dst_pin']]

            codes += "  assign %s = %s;\n" % (dst_pin_name, src_pin_name)

            if dst_pin_name in po_status:
                po_status[dst_pin_name] = True

        # fulfill empty POs
        for po_name, status in po_status.items():
            if not status:
                codes += "  assign %s = 1'b0;\n" % po_name

        codes += "endmodule\n"

        return codes
    
    def generate_full_adder_verilog(self, name: str) -> str:
        """
            Generate verilog code for the full adder
        """

        # previous version, DC use 2 FAs to synthesize, which is stupid
        # assign io_s = io_a ^ io_b ^ io_ci;
        # assign io_co = (io_a & io_b) | (io_b & io_ci) | (io_ci & io_a);

        codes = """
`ifndef OP_COMPTREE_FA
`define OP_COMPTREE_FA

module %s(
  input   io_a,
  input   io_b,
  input   io_ci,
  output  io_s,
  output  io_co
);
`ifdef DC
  DW01_add #(1) 
    adder ( .A(io_a), .B(io_b), .CI(io_ci), .SUM(io_s), .CO(io_co) );
`else
  wire  a_xor_b = io_a ^ io_b;
  wire  a_and_b = io_a & io_b;
  wire  a_and_cin = io_a & io_ci;
  wire  b_and_cin = io_b & io_ci;
  wire  _T_1 = a_and_b | b_and_cin;
  assign io_s = a_xor_b ^ io_ci;
  assign io_co = _T_1 | a_and_cin;
`endif
endmodule

`endif
""" % name
        
        return codes
    
    def generate_half_adder_verilog(self, name: str) -> str:
        """
            Generate verilog code for the half adder
        """

        codes = """
`ifndef OP_COMPTREE_HA
`define OP_COMPTREE_HA

module %s(
  input   io_a,
  input   io_b,
  output  io_s,
  output  io_co
);
  assign io_s = io_a ^ io_b;
  assign io_co = io_a & io_b;
endmodule

`endif
""" % name
        
        return codes
    
    def generate_fall_through_verilog(self, name: str) -> str:
        """
            Generate verilog code for the fall-through
        """

        codes = """
`ifndef OP_COMPTREE_FT
`define OP_COMPTREE_FT

module %s(
  input  io_a,
  output io_s
);
  assign io_s = io_a;
endmodule

`endif
""" % name
        
        return codes

    def mutate(self, column: int, stage: int, na_rank: int, nb_rank: int, na_pin: int, nb_pin: int) -> nx.DiGraph:
        """
            Swap input connection of two nodes at the same column & stage
            FIXME: use networkx grammer to handle properties
        """

        # duplicate a graph
        g = deepcopy(self.comp_graph)

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
        fa_pi_list = [PinType.A, PinType.B, PinType.CI]
        ha_pi_list = [PinType.A, PinType.B]

        if node_a.node_type == CompressorType.FA:
            if na_pin not in fa_pi_list:
                return None
        elif node_a.node_type == CompressorType.HA:
            if na_pin not in ha_pi_list:
                return None
        else:
            return None

        if node_b.node_type == CompressorType.FA:
            if nb_pin not in fa_pi_list:
                return None
        elif node_b.node_type == CompressorType.HA:
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
        if g.nodes[pred_a]['node_type'] == CompressorType.PI:
            g.edges[pred_a, node_b]['delay'] = 0
        if g.nodes[pred_b]['node_type'] == CompressorType.PI:
            g.edges[pred_b, node_a]['delay'] = 0

        return g
    
    def save(self, save_dir: str, dump_graph: bool = False):
        """
            Dump the data into the data_dir
        """

        mkdir(save_dir)

        stage_view = self.to_stage_view()
        count_view = stage_view.to_count_view()

        # dump ppgen configuration
        ppgen_config = self.pp_gen.get_config()
        ppgen_config_path = os.path.join(save_dir, 'ppgen_config.json')
        dump_json(ppgen_config, ppgen_config_path)

        # dump stage array
        stage_array_path = os.path.join(save_dir, 'stage_array.npy')
        np.save(stage_array_path, stage_view.stage_array)

        # dump count array
        count_array_path = os.path.join(save_dir, 'count_array.npy')
        np.save(count_array_path, count_view.count_array)

        # dump graph
        if dump_graph:
            graph_path = os.path.join(save_dir, 'graph.pkl')
            with open(graph_path, 'wb') as f:
                pkl.dump(self.comp_graph, f)
    
    # def annotate_delay(self):
    #     """
    #         Annotate the delay of each edge
    #     """
    #     annotate_delay(self.comp_graph, self.comp_estimator)

    # def get_critical_path(self) -> list:
    #     """
    #         Find the critical path of the graph
    #     """
    #     output_nodes = [n for n in self.comp_graph.nodes if n.node_type == CompressorType.PO]

    #     # find output nodes' predecessors with largest delay
    #     max_delay = 0
    #     max_node = None
    #     max_pred = None

    #     for node in output_nodes:
    #         for pred in self.comp_graph.predecessors(node):
    #             delay = self.comp_graph.edges[pred, node]['delay']
    #             if delay > max_delay:
    #                 max_delay = delay
    #                 max_node  = node
    #                 max_pred  = pred

    #     assert max_node is not None
    #     assert max_pred is not None

    #     # construct the critical path
    #     critical_path = []
        
    #     while max_pred.node_type != CompressorType.PI::
    #         critical_path.append(max_pred)

    #         # check critical edge type to find previous node
    #         edge_type = self.comp_graph.edges[max_pred, max_node]['src_pin']

    #         if edge_type == PinType.S:
    #             pred_edge_type = self.comp_graph.nodes[max_pred]['s_critical_transition']
    #         elif edge_type == PinType.CO:
    #             pred_edge_type = self.comp_graph.nodes[max_pred]['co_critical_transition']
    #         else:
    #             raise ValueError(f'Unknown edge type: {edge_type}')
            
    #         next_pred = [pred for pred in self.comp_graph.predecessors(max_pred) 
    #                         if self.comp_graph.edges[pred, max_pred]['dst_pin'] == pred_edge_type][0]
    #         max_pred, max_node = next_pred, max_pred

    #     return list(reversed(critical_path))
