from .base import PartialProduct, PartialProductGenerator
from typing import List

class AndMultPPG(PartialProductGenerator):
    """
        Partial product generator for AND-based multiplier
    """

    def __init__(self, multiplicand_bit: int, multiplier_bit: int, unsigned: bool = True) -> None:
        self.multiplicand_bit = multiplicand_bit
        self.multiplier_bit = multiplier_bit
        self.unsigned = unsigned

        # TODO: we only support Baugh-Wooley algorithm with m == n
        if not self.unsigned:
            assert self.multiplicand_bit == self.multiplier_bit

    def _get_init_pp_histogram(self) -> List[List[PartialProduct]]:
        if self.unsigned:
            return self._get_unsigned_init_pp_histogram()
        else:
            return self._get_signed_init_pp_histogram()
        
    def generate_verilog(self, module_name: str = 'AndMultPPG') -> str:
        m, n = self.multiplicand_bit, self.multiplier_bit

        # IO interface
        codes = "module %s(\n" % module_name
        for col, bits in enumerate(self.get_init_ppcnt_list()):
            if bits == 0: continue
            codes += "  output [%d:0] pp_%d,\n" % (bits-1, col)
        codes += "  input [%d:0] multiplicand,\n" % (m-1)
        codes += "  input [%d:0] multiplier\n" % (n-1)
        codes += ");\n"

        # generate PPs
        for pp in self.get_init_pp_list():
            codes += pp.generate_verilog()

        # endmodule
        codes += "endmodule\n"

        return codes

        
    def _get_unsigned_init_pp_histogram(self) -> List[List[PartialProduct]]:
        m, n = self.multiplicand_bit, self.multiplier_bit
        hist = [[] for _ in range(m + n)]

        for i in range(m):
            for j in range(n):
                pp = PartialProduct(assignment='multiplicand[%d] & multiplier[%d]' % (i, j))
                hist[i+j].append(pp)

        return hist
    
    def _get_signed_init_pp_histogram(self) -> List[List[PartialProduct]]:
        """
            Use modified Baugh-Wooley method
        """
        m, n = self.multiplicand_bit, self.multiplier_bit
        hist = [[] for _ in range(m + n)]

        # magic 1 at the first row of PPs
        hist[m].append(PartialProduct(assignment='1'))
        
        # magic 1 at the last row of PPs
        hist[m+n-1].append(PartialProduct(assignment='1'))

        # pp array
        for i in range(m):
            for j in range(n):
                if i == m - 1 and j == n - 1:
                    pp = PartialProduct(assignment='multiplicand[%d] & multiplier[%d]' % (i, j))
                elif i == m - 1 or j == n - 1:
                    pp = PartialProduct(assignment='~(multiplicand[%d] & multiplier[%d])' % (i, j))
                else:
                    pp = PartialProduct(assignment='multiplicand[%d] & multiplier[%d]' % (i, j))
                hist[i+j].append(pp)

        return hist
    
    def get_config(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'multiplicand_bit': self.multiplicand_bit,
            'multiplier_bit': self.multiplier_bit,
            'unsigned': self.unsigned
        }