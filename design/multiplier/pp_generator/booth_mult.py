from .base import PartialProduct, PartialProductGenerator
from typing import List
import math

class BoothRadix4MultPPG(PartialProductGenerator):
    """
        Partial product generator for Radix-4 Booth multiplier
    """

    def __init__(self, multiplicand_bit: int, multiplier_bit: int, unsigned: bool = True) -> None:
        self.multiplicand_bit = multiplicand_bit
        self.multiplier_bit = multiplier_bit
        self.unsigned = unsigned


    def _get_init_pp_histogram(self) -> List[List[PartialProduct]]:
        if self.unsigned:
            return self._get_unsigned_init_pp_histogram()
        else:
            return self._get_signed_init_pp_histogram()


    def generate_verilog(
        self, 
        module_name: str = 'BoothRadix4MultPPG',
    ) -> str:
        m, n = self.multiplicand_bit, self.multiplier_bit

        macro_name = 'OP_BOOTH4_ENCODER' if self.unsigned else 'OP_BOOTH4_ENCODER_SIGNED'

        # define booth encoder
        # For cases where Booth coefficient is negative, we need to take 2's complement
        # adding a sign bit for 2's complement is considered as adding a standalone partial product
        codes = """
`ifndef %s
`define %s

module Booth4Encoder #(
    parameter integer M = %d
)(
  input  [M-1:0] a_i,
  input  [2:0]   b_i,
  output [M:0]   booth_o,
  output         sign_o
);
    wire neg  = b_i[2];           // 100, 101, 110, 111  
    wire zero = &b_i | &(~b_i);   // 000, 111
    wire one  = b_i[1] ^ b_i[0];  // 001, 010, 101, 110
    // wire two  = (b_i == 3'b100) || (b_i == 3'b011);
    wire nonzero = |(b_i) ^ &(b_i);

    wire [M:0] a_extended = {%s, a_i};
    wire [M:0] a_shifted  = {a_i, 1'b0};

    // wire [M:0] booth_abs = one ? a_extended : (two ? a_shifted : {M+1{1'b0}});
    wire [M:0] booth_abs = (one ? a_extended : a_shifted) & {M+1{nonzero}};

    assign booth_o = {M+1{neg}} ^ booth_abs;
    assign sign_o  = neg;
                            
endmodule

`endif
""" % (macro_name, macro_name, m, "1'b0" if self.unsigned else "a_i[M-1]")

        # IO interface
        codes += "module %s(\n" % module_name
        for col, bits in enumerate(self.get_init_ppcnt_list()):
            if bits == 0: continue
            codes += "  output [%d:0] pp_%d,\n" % (bits-1, col)
        codes += "  input [%d:0] multiplicand,\n" % (m-1)
        codes += "  input [%d:0] multiplier\n" % (n-1)
        codes += ");\n"

        # extend multiplier (it's okay to extend more bits than necessary)
        codes += "  wire [%d:0] multiplier_ext = {%s, multiplier, 1'b0};\n" % \
            (n+2, "2'b0" if self.unsigned else "{2{multiplier[%d]}}" % (n-1))
        num_row = math.floor(n / 2) + 1 if self.unsigned else math.ceil(n / 2)

        # generate booth encoders
        for row in range(num_row):
            pp_bits = m + 1  # debug: send in full booth_o wire here, choose selectively in assign_pp
            booth_msb = 2*row+2
            booth_lsb = 2*row

            codes += "  wire [%d:0] booth_%d;\n" % (pp_bits-1, row)
            codes += "  wire sign_end_%d;\n" % row
            codes += """  Booth4Encoder #(
    .M(%d)
  ) booth4encoder_%d (
    .a_i(multiplicand), 
    .b_i(multiplier_ext[%d:%d]), 
    .booth_o(booth_%d), 
    .sign_o(sign_end_%d)
  );
""" % (m, row, booth_msb, booth_lsb, row, row)

        # generate PPs
        for pp in self.get_init_pp_list():
            codes += pp.generate_verilog()

        # endmodule
        codes += "endmodule\n"

        return codes


    def _get_unsigned_init_pp_histogram(self):
        m, n = self.multiplicand_bit, self.multiplier_bit
        hist = [[] for _ in range(m + n)]

        def assign_pp(col: int, assignment: str):
            nonlocal hist
            if col >= len(hist):
                return
            pp = PartialProduct(assignment=assignment)
            hist[col].append(pp)

        num_row = math.floor(n / 2) + 1

        for row in range(num_row):
            # assign partial products
            pp_bits = m + 1
            for i in range(pp_bits):
                col = i + 2 * row
                assign_pp(col, "booth_%d[%d]" % (row, i))

            # assign sign bits
            if row == 0:
                assign_pp(pp_bits, "sign_end_0")
                assign_pp(pp_bits+1, "sign_end_0")
                assign_pp(pp_bits+2, "~sign_end_0")
            elif row < num_row - 1:
                assign_pp(2*row+pp_bits, "~sign_end_%d" % row)
                assign_pp(2*row+pp_bits+1, "1'b1")
            else:
                continue

            # assign the ending sign bit for handling 2's complement
            if row < num_row - 1:
                assign_pp(2*row, "sign_end_%d" % row)

        return hist
        

    def _get_signed_init_pp_histogram(self):
        m, n = self.multiplicand_bit, self.multiplier_bit
        hist = [[] for _ in range(m + n)]

        def assign_pp(col: int, assignment: str):
            nonlocal hist
            if col >= len(hist):
                return
            pp = PartialProduct(assignment=assignment)
            hist[col].append(pp)

        num_row = math.ceil(n / 2)

        for row in range(num_row):
            # assign partial products
            pp_bits = m + 1
            for i in range(pp_bits):
                col = i + 2 * row
                if i == pp_bits - 1:
                    # ~x = 1-x  -> -x = ~x + (-1)
                    assign_pp(col, "~booth_%d[%d]" % (row, i))
                else:
                    assign_pp(col, "booth_%d[%d]" % (row, i))

            # assign sign bits
            if row == 0:
                assign_pp(pp_bits-1, "1'b1")
                assign_pp(pp_bits, "1'b1")
            else:
                assign_pp(2*row+pp_bits, "1'b1")

            # assign the ending sign bit for handling 2's complement
            if self.unsigned:
                if row < num_row - 1:
                    assign_pp(2*row, "sign_end_%d" % row)
            else:
                assign_pp(2*row, "sign_end_%d" % row)

        return hist
    
    def get_config(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'multiplicand_bit': self.multiplicand_bit,
            'multiplier_bit': self.multiplier_bit,
            'unsigned': self.unsigned
        }