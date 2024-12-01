import os
import random

from .base import CompTreeTopModule

from design.multiplier.pp_generator import PartialProductGenerator, FusedMacPPG
from design.multiplier.comp_tree import CompTreeGraphView
from design.adder.add_config import AdderConfig
from utils import mkdir, execute, remove

class MacTop(CompTreeTopModule):
    """
        Top module for fused-MAC
    """

    def __init__(self, comp_tree: CompTreeGraphView, final_adder: AdderConfig) -> None:
        super().__init__(comp_tree, final_adder)
        assert isinstance(comp_tree.pp_gen, FusedMacPPG)

    @property
    def unsigned(self) -> bool:
        return self.pp_gen.unsigned

    def generate_verilog(
        self, 
        top_module_name: str = 'CompTreeTopModule', 
        ppg_name: str = 'PartialProductGenerator', 
        ct_name: str = 'CompressorTree', 
        cpa_name: str = 'CarryPropagateAdder'
    ) -> str:
        ppg_codes = self.pp_gen.generate_verilog(ppg_name)
        ct_codes = self.comp_tree.generate_verilog(ct_name)
        cpa_codes = self.final_adder.generate_verilog(cpa_name)

        codes = ppg_codes + ct_codes + cpa_codes

        m, n = self.pp_gen.multiplicand_bit, self.pp_gen.multiplier_bit

        codes += """
module %s (
  input clock,
  input reset,
  input [%d:0] multiplicand,
  input [%d:0] multiplier,
  input [%d:0] accumulator,
  output [%d:0] product
);
""" % (
    top_module_name,
    m - 1,
    n - 1,
    m + n - 1,
    m + n - 1,
)
        
        # wire for intermediate signals
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "  wire [%d:0] pp_%d;\n" % (total_pp_bits-1, col)
        codes += "  wire [%d:0] augend;\n" % (m+n-1)
        codes += "  wire [%d:0] addend;\n\n" % (m+n-1)

        # partial product generator
        codes += "  %s pp_gen(\n" % ppg_name
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "    .pp_%d(pp_%d),\n" % (col, col)
        codes += "    .multiplicand(multiplicand),\n"
        codes += "    .multiplier(multiplier),\n"
        codes += "    .accumulator(accumulator)\n"
        codes += "  );\n\n"

        # compressor tree
        codes += "  %s comp_tree(\n" % ct_name
        for col, total_pp_bits in enumerate(self.pp_gen.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "    .io_pp_%d(pp_%d),\n" % (col, col)
        codes += "    .io_augend(augend),\n"
        codes += "    .io_addend(addend)\n"
        codes += "  );\n\n"

        # final adder
        codes += "  %s final_adder(\n" % cpa_name
        codes += "    .io_augend(augend),\n"
        codes += "    .io_addend(addend),\n"
        codes += "    .io_outs(product)\n"
        codes += "  );\n\n"

        codes += "endmodule\n"

        return codes
    
    def generate_testbench(
        self,
        top_module_name: str = 'CompTreeTopModule',
        testbench_name: str = 'CompTreeTopModuleTestbench',
        n_testcase: int = 100,
        random_seed: int = 42,
    ) -> str:
        rng = random.Random(random_seed)
        m, n = self.pp_gen.multiplicand_bit, self.pp_gen.multiplier_bit
        prefix = '' if self.unsigned else 'signed'

        codes = f"""
`timescale 1ns / 1ps

module {testbench_name};

// Testbench signals
reg {prefix} [{m-1}:0] a;
reg {prefix} [{n-1}:0] b;
reg {prefix} [{m+n-1}:0] c;
wire {prefix} [{m+n-1}:0] d;
wire {prefix} [{m+n-1}:0] gold;

// Instantiate the multiplier module
{top_module_name} mult (
    .multiplicand(a),
    .multiplier(b),
    .accumulator(c),
    .product(d)
);

assign gold = a * b + c;

// Test procedure
initial begin
    $display("Testbench starts...");
    // Initialize inputs
    a = 0; b = 0;
    #10;
"""
        for i in range(n_testcase):
            if self.unsigned:
                a_val = rng.randint(0, 2 ** m - 1)
                b_val = rng.randint(0, 2 ** n - 1)
                c_val = rng.randint(0, 2 ** (m+n) - 1)
            else:
                a_val = rng.randint(-(2 ** (m-1)), 2 ** (m-1) - 1)
                b_val = rng.randint(-(2 ** (n-1)), 2 ** (n-1) - 1)
                c_val = rng.randint(-(2 ** (m+n-1)), 2 ** (m+n-1) - 1)

            codes += f"""
    // Test case {i}
    a = {'-' if self.unsigned is False and a_val < 0 else ''}{m}'{'' if self.unsigned else 's'}d{abs(a_val)}; 
    b = {'-' if self.unsigned is False and b_val < 0 else ''}{n}'{'' if self.unsigned else 's'}d{abs(b_val)};
    c = {'-' if self.unsigned is False and c_val < 0 else ''}{m+n}'{'' if self.unsigned else 's'}d{abs(c_val)};
    #10;
    if (d !== gold) begin
        $display("a=%d, b=%d, c=%d, d=%b, gold=%b", a, b, c, d, gold);
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