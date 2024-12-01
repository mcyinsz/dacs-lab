from .base import PartialProduct, PartialProductGenerator
from .and_mult import AndMultPPG
from .booth_mult import BoothRadix4MultPPG
from typing import List

class FusedMacPPG(PartialProductGenerator):
    """
        Fused MAC partial product generator, basiclly a wrapper module with one more row of PP
        Currently we only support acc_width == m + n (it's difficult to handle signed case)
    """
    def __init__(self, mult_ppg: PartialProductGenerator):
        self.mult_ppg = mult_ppg
        assert isinstance(self.mult_ppg, (AndMultPPG, BoothRadix4MultPPG))

    def _get_init_pp_histogram(self) -> List[List[PartialProduct]]:
        mult_ppcnt_list = self.mult_ppg.get_init_ppcnt_list()
        hist = [[] for _ in range(len(mult_ppcnt_list))]

        for col, bits in enumerate(mult_ppcnt_list):
            for b in range(bits):
                pp = PartialProduct(assignment='mult_pp_%d[%d]' % (col, b))
                hist[col].append(pp)
            acc_pp = PartialProduct(assignment='accumulator[%d]' % col)
            hist[col].append(acc_pp)

        return hist

    def generate_verilog(self, module_name: str = 'FusedMacPPG', mult_ppg_name: str = 'MultPPG') -> str:
        m, n = self.mult_ppg.multiplicand_bit, self.mult_ppg.multiplier_bit

        # internal module
        codes = self.mult_ppg.generate_verilog(module_name=mult_ppg_name)

        # wrapper module interface
        codes += "module %s(\n" % module_name
        for col, bits in enumerate(self.get_init_ppcnt_list()):
            if bits == 0: continue
            codes += "  output [%d:0] pp_%d,\n" % (bits-1, col)
        codes += "  input [%d:0] multiplicand,\n" % (m-1)
        codes += "  input [%d:0] multiplier,\n" % (n-1)
        codes += "  input [%d:0] accumulator\n" % (m+n-1)
        codes += ");\n"

        # generate PPs
        for col, bits in enumerate(self.mult_ppg.get_init_ppcnt_list()):
            if bits == 0: continue
            codes += "  wire [%d:0] mult_pp_%d;\n" % (bits-1, col)

        codes += "  %s mult_ppg(\n" % mult_ppg_name
        for col, bits in enumerate(self.mult_ppg.get_init_ppcnt_list()):
            if bits == 0: continue
            codes += "    .pp_%d(mult_pp_%d),\n" % (col, col)
        codes += "    .multiplicand(multiplicand),\n"
        codes += "    .multiplier(multiplier)\n"
        codes += "  );\n"

        for pp in self.get_init_pp_list():
            codes += pp.generate_verilog()

        # endmodule
        codes += "endmodule\n"

        return codes
    
    @property
    def multiplicand_bit(self) -> int:
        return self.mult_ppg.multiplicand_bit
    
    @property
    def multiplier_bit(self) -> int:
        return self.mult_ppg.multiplier_bit
    
    @property
    def unsigned(self) -> bool:
        return self.mult_ppg.unsigned
    
    def get_config(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'multiplicand_bit': self.multiplicand_bit,
            'multiplier_bit': self.multiplier_bit,
            'unsigned': self.unsigned
        }