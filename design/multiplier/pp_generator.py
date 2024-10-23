import abc
import math

class PartialProductGenerator(abc.ABC):
    """
        Abstract class for partial product generator of multipliers
    """
    
    def __init__(self, m, n):
        self.m = m
        self.n = n

    @abc.abstractmethod
    def get_init_ppcnt_list(self) -> list:
        """
            Get initial number of partial products
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_verilog(self) -> str:
        """
            Generate verilog codes for partial product generator
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_num_row(self) -> int:
        raise NotImplementedError()
    
    # helper functions
    def generate_io_interface(self) -> str:
        """
            Generate verilog codes defining module name and IO
        """
        codes = "module PartialProductGenerator(\n"
        for col, total_pp_bits in enumerate(self.get_init_ppcnt_list()):
            if total_pp_bits == 0: continue
            codes += "  output [%d:0] pp_%d,\n" % (total_pp_bits-1, col)
        codes += "  input [%d:0] multiplicand,\n" % (self.m-1)
        codes += "  input [%d:0] multiplier\n" % (self.n-1)
        codes += ");\n"
        return codes
    

class AndPPGenerator(PartialProductGenerator):
    """
        Partial product generator using AND gates
        By default for unsigned multiplier
    """
    
    def __init__(self, m, n):
        super().__init__(m, n)

    def get_init_ppcnt_list(self) -> list:
        m, n = self.m, self.n
        ppcnt_list = [0 for _ in range(m + n)]

        for i in range(n):
            for j in range(m):
                ppcnt_list[i+j] += 1

        return ppcnt_list

    def generate_verilog(self) -> str:
        m, n = self.m, self.n
        pp_rank_list = [0 for _ in range(m + n)]

        codes = self.generate_io_interface()

        for i in range(n):
            for j in range(m):
                col = i + j
                rank = pp_rank_list[col]
                codes += "  assign pp_%d[%d] = multiplicand[%d] & multiplier[%d];\n" % (col, rank, j, i)
                pp_rank_list[col] += 1

        codes += "endmodule\n"

        assert pp_rank_list == self.get_init_ppcnt_list(), "Rank list mismatch"

        return codes
    
    def get_num_row(self) -> int:
        return self.n
    

class UnsignedBoothRadix4PPGenerator(PartialProductGenerator):
    """
        Unsigned partial product generator using Radix-4 Booth's algorithm
    """

    def __init__(self, m, n):
        super().__init__(m, n)
        assert n % 2 == 0, "only supports even number of multiplier bits"

    def get_init_ppcnt_list(self) -> list:
        m, n = self.m, self.n
        num_row = n // 2 + 1
        pp_list = [0 for _ in range(m + n)]

        # booth-based partial products
        for row in range(num_row):
            pp_bits = m + 1 if row < num_row - 1 else m
            for j in range(pp_bits):
                pp_list[2*row+j] += 1

        # sign for 2's complement
        for row in range(num_row-1):
            pp_list[2*row] += 1

        # sign extension 1
        for row in range(num_row-1):
            pp_bits = m + 1 if row < num_row - 1 else m
            pp_list[2*row + pp_bits] += 1

        # sign extension 2
        for row in range(num_row-2):
            pp_bits = m + 1 if row < num_row - 1 else m
            pp_list[2*row + pp_bits + 1] += 1

        # sign extension 3
        pp_list[m + 3] += 1

        return pp_list

    def generate_verilog(self) -> str:
        m, n = self.m, self.n
        num_row = n // 2 + 1
        pp_rank_list = [0 for _ in range(m + n)]

        def assign_pp(col, wire):
            nonlocal pp_rank_list, codes
            rank = pp_rank_list[col]
            codes += f"  assign pp_{col}[{rank}] = {wire};\n"
            pp_rank_list[col] += 1

        # prepare booth encoder
        # For cases where Booth coefficient is negative, we need to take 2's complement
        # the sign bit to add is considered a standalone partial product
        codes = """
module Booth4Encoder (
  input  [%d:0]  a_i,
  input  [2:0]   b_i,
  output [%d:0]  booth_o,
  output         sign_o
);
    wire neg  = b_i[2];
    wire zero = (b_i == 3'b000) || (b_i == 3'b111);
    wire two  = (b_i == 3'b100) || (b_i == 3'b011);
    wire one  = !zero & !two;

    wire [%d:0] a_extended = {a_i[%d], a_i};
    wire [%d:0] a_shifted  = {a_i, 1'b0};

    wire [%d:0] booth_abs = one  ? a_extended : 
                     two  ? a_shifted : 
                            %d'b0;
    assign booth_o = neg ? ~booth_abs : booth_abs;
    assign sign_o  = neg & ~(b_i[1] & b_i[0]);
                            
endmodule
""" % (
    m-1, m,            # interface
    m, m-1,            # a_extended
    m,                 # a_shifted
    m, m,              # booth_abs
)

        codes += self.generate_io_interface()
        
        # extend multiplier
        codes += "  wire [%d:0] multiplier_ext = {2'b0, multiplier, 1'b0};\n" % (n+3)
        
        # booth encoder for partial products
        for row in range(num_row):
            pp_bits = m + 1  # debug: send in full booth_o wire here, choose selectively in assign_pp
            booth_msb = 2*row+2
            booth_lsb = 2*row

            codes += "  wire [%d:0] booth_%d;\n" % (pp_bits-1, row)
            codes += "  wire sign_end_%d;\n" % row
            codes += """  Booth4Encoder booth4encoder_%d(
    .a_i(multiplicand), 
    .b_i(multiplier_ext[%d:%d]), 
    .booth_o(booth_%d), 
    .sign_o(sign_end_%d)
  );
""" % (row, booth_msb, booth_lsb, row, row)

        for row in range(num_row):
            # assign partial products
            pp_bits = m + 1 if row < num_row - 1 else m
            for i in range(pp_bits):
                col = i + 2 * row
                assign_pp(col, "booth_%d[%d]" % (row, i))

            # assign sign bits
            if row == 0:
                assign_pp(2*row+pp_bits, "sign_end_0")
                assign_pp(2*row+pp_bits+1, "sign_end_0")
                assign_pp(2*row+pp_bits+2, "~sign_end_0")

            elif row < num_row - 2:
                assign_pp(2*row+pp_bits, "~sign_end_%d" % row)
                assign_pp(2*row+pp_bits+1, "1'b1")
            elif row < num_row - 1:
                assign_pp(2*row+pp_bits, "~sign_end_%d" % row)
            else:
                continue

            # assign the ending sign bit for handling 2's complement
            if row < num_row - 1:
                assign_pp(2*row, "sign_end_%d" % row)
            
        codes += "endmodule\n"

        assert pp_rank_list == self.get_init_ppcnt_list(), "Rank list mismatch"
        
        return codes
    
    def get_num_row(self) -> int:
        return self.n // 2 + 1

if __name__ == '__main__':
    # debug init pp list
    for n_bit in [8, 16]:
        print(f"n_bit = {n_bit}")
        # pp_gen = UnsignedBoothRadix4PPGenerator(n_bit, n_bit)
        pp_gen = AndPPGenerator(n_bit, n_bit)
        print(pp_gen.get_init_ppcnt_list())
        print(pp_gen.generate_verilog())
        print()