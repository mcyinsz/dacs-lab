import numpy as np
import copy
import os

from utils import mkdir, create_hash

class PPAdderConfig():
    
    def __init__(self,
                 input_bit: int,
                 required_mat: np.ndarray = None,
                 ) -> None:
        """init the adder with N bits

        Args:
            input_bit (int): N
        """
        self.N = input_bit
        if required_mat is not None:
            assert required_mat.shape == (self.N, self.N)
            self.min_mat = np.zeros((self.N, self.N), dtype=bool)
            self.level_mat = np.zeros((self.N, self.N), dtype=int)
            self.node_mat = required_mat
            self._legalize()
            self._legalize_min()
        else:
            self.min_mat = np.zeros((self.N, self.N), dtype=bool)
            self.node_mat = np.zeros((self.N, self.N), dtype=bool)
            self.level_mat = np.zeros((self.N, self.N), dtype=int)
            self.legalize()

    def add(self, msb, lsb) -> bool:
        """add a node with (msb, lsb) to the node matrix

        Args:
            msb (int): indicates row of the mat
            lsb (int): indicates column of the mat
        """
        if self.node_mat[msb, lsb] == 1:
            return False

        self.min_mat[msb, lsb] = 1
        last_l = msb
        for l in range(msb-1, -1, -1):
            if self.min_mat[msb, l] == 1:
                # delete lp(msb, l) from minlist
                self.min_mat[last_l-1, l] = 0
                last_l = l
        self.legalize()
        return True
    
    def delete(self, msb, lsb) -> bool:
        """delete a node with (msb, lsb) from the node matrix

        Args:
            msb (int): indicates row of the mat
            lsb (int): indicates column of the mat
        """
        if self.min_mat[msb, lsb] == 0:
            return False
        
        self.min_mat[msb, lsb] = 0
        self.legalize()
        return True

    def _legalize(self) -> None:
        """legalize the node_mat
        """
        min_l = self.N # record the minimal l to update level matrix
        for m in range(self.N-1, -1, -1):
            last_l = m
            for l in range(m-1, -1, -1):
                # if (m, l) is in nodelist, add lp(m, l) to nodelist
                if self.node_mat[m, l] == 1:
                    self.node_mat[last_l-1, l] = 1
                    min_l = last_l - 1 if (last_l -1 ) < min_l else min_l
                    last_l = l
        self._update_level(min_l)

    def _legalize_min(self) -> None:
        """construct a legalized min_mat from node_mat
        """
        self.min_mat = copy.deepcopy(self.node_mat)
        for m in range(0, self.N):
            self.min_mat[m, m] = 0
            self.min_mat[m, 0] = 0
            last_l = m
            for l in range(m-1, 0, -1):
                if self.min_mat[m, l] == 1:
                    # delete lp(msb, l) from minlist
                    self.min_mat[last_l-1, l] = 0
                    last_l = l

    def legalize(self) -> None:
        """legalize the adder based on min_mat
        """
        self.node_mat = copy.deepcopy(self.min_mat)

        for m in range(self.N):
            self.node_mat[m, 0] = 1
            self.node_mat[m, m] = 1

        self._legalize()

    def _update_level(self, change_node=0) -> None:
        """update the level matrix of the adder through a bottom-up scheme

        Args:
            change_node (int, optional): The node changed by the action. Only need to change levels that greater than the changed node. Defaults to 0.
        """
        for m in range(change_node, self.N):
            self.level_mat[m, :] = 0
            self.level_mat[m, m] = 1
            last_l = m
            for l in range(m-1, -1, -1):
                if self.node_mat[m, l]:
                    self.level_mat[m, l] = max(self.level_mat[m, last_l], self.level_mat[last_l-1, l]) + 1
                    last_l = l
    
    def _ml_to_idx(self, m, l) -> int:
        """given msb and lsb of the node, output the idx of the node in the adjacent list
        """
        return int(self.N*(self.N+1)/2 - m*(m+1)/2 - l - 1)

    def _construct_graph(self):
        """return an adjacent list
        returns:
            List(dict): a list of node contains the in nodes and out nodes
        
        """
        # init adjacent list
        graph = [{"in": [], "out": []} for _ in range(self._ml_to_idx(0, 0) + 1)]
        
        for m in range(self.N-1, -1, -1):
            last_l = m
            for l in range(m-1, -1, -1):
                if self.node_mat[m, l] == 1:
                    idx = self._ml_to_idx(m, l)
                    # lower parent
                    l_idx = self._ml_to_idx(last_l-1, l)
                    graph[idx]["in"].append(l_idx)
                    graph[l_idx]["out"].append(idx)
                    # upper parent
                    u_idx = self._ml_to_idx(m, last_l)
                    graph[idx]["in"].append(u_idx)
                    graph[u_idx]["out"].append(idx)
                    # update last_l
                    last_l = l
        return graph

    def generate_verilog(self) -> str:
        """
            Generate verilog code for the graph
        """
        N = self.N

        codes = """
module PG(
  input   io_i_a,
  input   io_i_b,
  output  io_o_p,
  output  io_o_g
);
  assign io_o_p = io_i_a ^ io_i_b;
  assign io_o_g = io_i_a & io_i_b;
endmodule

module Black(
  input   io_i_pj,
  input   io_i_gj,
  input   io_i_pk,
  input   io_i_gk,
  output  io_o_g,
  output  io_o_p
);
  wire  _T = io_i_gj & io_i_pk;
  assign io_o_g = io_i_gk | _T;
  assign io_o_p = io_i_pk & io_i_pj;
endmodule
""" # copied from EasyMAC compiled verilog

        codes += """
module PPAdder(
  input        clock,
  input        reset,
  input  [%d:0] io_augend,
  input  [%d:0] io_addend,
  output [%d:0] io_outs
);
""" % (N-1, N-1, N-1)
        graph = self._construct_graph()
        for idx, node in enumerate(graph):
            if len(node["in"]) == 0 and len(node["out"]) == 0: continue # the node doesn't exist
            elif len(node["in"]) == 0: # input node
                codes += """
  wire  PG_%d_io_i_a;
  wire  PG_%d_io_i_b;
  wire  PG_%d_io_o_p;
  wire  PG_%d_io_o_g;
  PG PG_%d (
    .io_i_a(PG_%d_io_i_a),
    .io_i_b(PG_%d_io_i_b),
    .io_o_p(PG_%d_io_o_p),
    .io_o_g(PG_%d_io_o_g)
  );
""" % (idx, idx, idx, idx, idx, idx, idx, idx, idx)
            else: # black node
                codes += """
  wire  Black_%d_io_i_pj;
  wire  Black_%d_io_i_gj;
  wire  Black_%d_io_i_pk;
  wire  Black_%d_io_i_gk;
  wire  Black_%d_io_o_g;
  wire  Black_%d_io_o_p;
  Black Black_%d (
    .io_i_pj(Black_%d_io_i_pj),
    .io_i_gj(Black_%d_io_i_gj),
    .io_i_pk(Black_%d_io_i_pk),
    .io_i_gk(Black_%d_io_i_gk),
    .io_o_g(Black_%d_io_o_g),
    .io_o_p(Black_%d_io_o_p)
  );
""" % (idx, idx, idx, idx, idx, idx, idx, idx, idx, idx, idx, idx, idx)
        
        out_str = "};\n"
        for idx in range(N):
            pg_idx = self._ml_to_idx(idx, idx)
            if idx == 0:
                codes += """
  wire  res_%d = PG_%d_io_o_p;
""" % (idx, pg_idx)
            elif idx == 1:
                idx_0 = self._ml_to_idx(0, 0)
                codes += """
  wire  res_%d = PG_%d_io_o_p ^ PG_%d_io_o_g;
""" % (idx, pg_idx, idx_0)
            else:
                black_idx = self._ml_to_idx(idx-1, 0)
                codes += """
  wire  res_%d = PG_%d_io_o_p ^ Black_%d_io_o_g;
""" % (idx, pg_idx, black_idx)
            if idx == N-1:
                out_str = "assign io_outs = {res_%d" % (idx) + out_str
            else:
                out_str = ", res_%d" % (idx) + out_str

        codes += out_str
        for idx in range(N):
            pg_idx = self._ml_to_idx(idx, idx)
            codes += f"""
  assign PG_%d_io_i_a = io_augend[%d];
  assign PG_%d_io_i_b = io_addend[%d];
""" % (pg_idx, idx, pg_idx, idx)

        for idx, node in enumerate(graph):
            if len(node["in"]) != 0:
                j = max(node["in"])
                k = min(node["in"])
                node_j = graph[j]
                str_j = f"Black_{j}_io_o" if len(node_j["in"]) != 0 else f"PG_{j}_io_o"
                node_k = graph[k]
                str_k = f"Black_{k}_io_o" if len(node_k["in"]) != 0 else f"PG_{k}_io_o"
                codes += """
  assign Black_%d_io_i_pj = %s_p;
  assign Black_%d_io_i_gj = %s_g;
  assign Black_%d_io_i_pk = %s_p;
  assign Black_%d_io_i_gk = %s_g;
""" % (idx, str_j, idx, str_j, idx, str_k, idx, str_k)
        
        codes += "\nendmodule"
        return codes
    
    @property
    def hash(self):
        s = str(tuple(np.reshape(self.node_mat, -1)))
        return create_hash(s)
    
    @property
    def level(self) -> int:
        """return the max level of the current adder

        Returns:
            int: level of the adder
        """
        return self.level_mat.max()

    @property
    def area(self) -> int:
        """return the number of logic gates of the current adder. This is just a rough estimate of the area.

        Returns:
            int: number of gates of the adder
        """
        return self.node_mat.sum() - self.N

    def dump_data(self, save_dir: str):
        """
            Dump the data into the data_dir
        """

        mkdir(save_dir)

        # dump encoded data
        encoded_data_path = os.path.join(save_dir, 'data.npy')
        np.save(encoded_data_path, self.node_mat)

        # dump verilog
        verilog_path = os.path.join(save_dir, 'ppadder.v')
        with open(verilog_path, 'w') as f:
            f.write(self.generate_verilog())
