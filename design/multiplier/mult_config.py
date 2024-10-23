import os
import random

from .pp_generator import PartialProductGenerator
from .comptree_graph_view import CompTreeGraphView
from ..adder.add_config import PPAdderConfig

from utils import mkdir

class MultiplierConfig():
    """
        Construct a multiplier with Partial Product Generator, CompressorTree and Final Full Adder
    """

    def __init__(
        self,
        m,
        n,
        pp_generator: PartialProductGenerator,
        comp_graph: CompTreeGraphView,
        final_adder: PPAdderConfig,
    ) -> None:
        self.m = m
        self.n = n
        self.pp_gen = pp_generator
        self.comp_graph = comp_graph
        self.final_adder = final_adder

        # if we use non-default adder, check bit-width
        if final_adder is not None:
            assert m + n == final_adder.N

        # if we use non-default multiplier, check bit-width
        if (pp_generator is None) and (comp_graph is None):
            pass
        elif (pp_generator is not None) and (comp_graph is not None):
            assert m == comp_graph.m
            assert n == comp_graph.n
            assert m == pp_generator.m
            assert n == pp_generator.n
            assert type(pp_generator).__name__ == type(comp_graph.pp_gen).__name__
        else:
            raise ValueError("Invalid multiplier configuration")

    def generate_verilog(self) -> str:
        """
            Generate verilog codes for multiplier
        """
        m, n = self.m, self.n

        # case-1: use default multiplier
        if (self.pp_gen is None) and (self.comp_graph is None):
            codes = """
module Multiplier(
  input           clock,
  input           reset,
  input [%d:0]    multiplicand,
  input [%d:0]    multiplier,
  output [%d:0]   product
);
  assign product = multiplicand * multiplier;
endmodule
""" % (m-1, n-1, m+n-1)
            return codes

        # case-2: use non-default multiplier
        codes = self.pp_gen.generate_verilog() + self.comp_graph.generate_verilog()
        if self.final_adder is not None:
            codes += self.final_adder.generate_verilog()

        # io interface
        codes += """
module Multiplier(
  input           clock,
  input           reset,
  input [%d:0]    multiplicand,
  input [%d:0]    multiplier,
  output [%d:0]   product
);
""" % (m-1, n-1, m+n-1)
        
        # wire for intermediate signals
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "  wire [%d:0] pp_%d;\n" % (total_pp_bits-1, col)
        codes += "  wire [%d:0] augend;\n" % (m+n-1)
        codes += "  wire [%d:0] addend;\n\n" % (m+n-1)

        # partial product generator
        codes += "  PartialProductGenerator pp_gen(\n"
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "    .pp_%d(pp_%d),\n" % (col, col)
        codes += "    .multiplicand(multiplicand),\n"
        codes += "    .multiplier(multiplier)\n"
        codes += "  );\n\n"

        # compressor tree
        codes += "  CompressorTree comp_tree(\n"
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "    .io_pp_%d(pp_%d),\n" % (col, col)
        codes += "    .io_augend(augend),\n"
        codes += "    .io_addend(addend)\n"
        codes += "  );\n\n"

        # final adder
        if self.final_adder is not None:
            codes += """
PPAdder final_adder(
    .io_augend(augend),
    .io_addend(addend),
    .io_outs(product)
);
endmodule
"""
        else:
            codes += "  assign product = augend + addend;\n"
            codes += "endmodule\n"

        return codes
    
    def generate_testbench(self, testcase=10, seed=42) -> str:
        """
            Generate testbench for multiplier
        """
        rng = random.Random(seed)
        m, n = self.m, self.n

        codes = f"""
`timescale 1ns / 1ps

module testbench;

// Testbench signals
reg [{m-1}:0] a;
reg [{n-1}:0] b;
wire [{m+n-1}:0] c;
wire [{m+n-1}:0] gold;

// Instantiate the multiplier module
Multiplier mult (
    .multiplicand(a),
    .multiplier(b),
    .product(c)
);

assign gold = a * b;

// Test procedure
initial begin
    $display("Testbench starts...");
    // Initialize inputs
    a = 0; b = 0;
    #10;
"""
        for i in range(testcase):
            a_val = rng.randint(0, 2 ** m - 1)
            b_val = rng.randint(0, 2 ** n - 1)

            codes += f"""
    // Test case {i}
    a = {a_val}; b = {b_val};
    #10;
    if (c !== gold) begin
        $display("a=%d, b=%d, c=%d, gold=%d", a, b, c, gold);
        $fatal(1);
    end
"""
        codes += """
    // End simulation
    $display("Testbench finished!");
    $finish;
end

endmodule
"""
        return codes

        
    def dump_data(self, save_dir: str, dump_mult_graph: bool = False):
        """
            Dump multiplier configuration data to save_dir
        """
        mkdir(save_dir)

        # dump compressor tree config
        if self.comp_graph is not None:
            self.comp_graph.dump_data(save_dir, dump_graph=dump_mult_graph)

        # dump final adder config
        if self.final_adder is not None:
            self.final_adder.dump_data(save_dir)

        # dump verilog codes
        verilog_path = os.path.join(save_dir, "multiplier.v")
        with open(verilog_path, "w") as f:
            f.write(self.generate_verilog())

        # dump testbench codes
        testbench_path = os.path.join(save_dir, "testbench.v")
        with open(testbench_path, "w") as f:
            f.write(self.generate_testbench())