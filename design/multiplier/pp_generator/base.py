import abc
from dataclasses import dataclass
from typing import List
import math

@dataclass
class PartialProduct():
    """
        Dataclass to easily manipulate generated partial products
    """
    # positional information of PP
    column: int = None
    rank: int = None

    # in most cases, PP can be directly generated from PIs
    # in rare cases (e.g. Booth multiplier), you need to define other temporary signals
    assignment: str = None

    def generate_verilog(self) -> str:
        """
            Generate verilog code for partial product
        """
        assert self.column is not None
        assert self.rank is not None
        assert self.assignment is not None
        return f"  assign pp_{self.column}[{self.rank}] = {self.assignment};\n"



class PartialProductGenerator(abc.ABC):
    """
        Abstract class for partial product generator of multipliers
    """

    @abc.abstractmethod
    def _get_init_pp_histogram(self) -> List[List[PartialProduct]]:
        """
            Get initial histogram of partial products
            Return:
                A list of length L (number of columns),
                where each element is a list of partial products in a specific column.
                The column and rank will be automatically assigned by interface function.
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def generate_verilog(self, module_name: str = None) -> str:
        """
            Generate verilog codes for partial product generator
        """
        raise NotImplementedError()


    @abc.abstractmethod
    def get_config(self) -> dict:
        """
            PPG configuration in readable format
        """
        raise NotImplementedError()
    

    # helper functions


    def get_init_pp_histogram(self) -> List[List[PartialProduct]]:
        """
            Get initial histogram of partial products
            Return:
                A list of length L (number of columns),
                where each element is a list of partial products in a specific column.
        """
        hist = self._get_init_pp_histogram()
        for col, pps in enumerate(hist):
            for rank, pp in enumerate(pps):
                pp.column = col
                pp.rank = rank

        return hist

    
    @property
    def num_column(self) -> int:
        """
            Get number of columns (should be identical to comptree input width)
        """
        return len(self.get_init_pp_histogram())
    

    @property
    def num_min_stage(self) -> int:
        """
            Get minimum reduction stage with Dadda multiplier's derivation.
            maximize j s.t. d_j < min(m, n) <= d_{j+1}
            https://en.wikipedia.org/wiki/Dadda_multiplier
        """
        n = max(self.get_init_ppcnt_list())
        assert n <= 256
        d, i = 2, 1
        while True:
            d_next = math.floor(1.5 * d)
            if d < n and d_next >= n:
                return i
            d = d_next
            i += 1
        

    def get_init_pp_list(self) -> List[PartialProduct]:
        """
            Get initial list of partial products by flattening the histogram
        """
        return sum(self.get_init_pp_histogram(), [])
    
    
    def get_init_ppcnt_list(self) -> List[int]:
        """
            Get initial number of partial products
        """
        return [len(pps) for pps in self.get_init_pp_histogram()]
    
    def to_dict(self) -> dict:
        return self.__dict__