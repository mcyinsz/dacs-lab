from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np

class CompressorType(Enum):
    """
        Enum for compressor types
    """
    FA = 1  # full adder
    HA = 2  # half adder
    FT = 3  # fall-through
    PI = 4  # initial PP is deemed as a special compressor at stage -1
    PO = 5  # outputs to final adder is also deemed as a special compressor at stage S


@dataclass
class Compressor:
    """
        Compressor data structure, serves as node in comptree graph view
    """
    column: int
    stage:  int 
    rank:   int
    node_type: CompressorType

    def __str__(self) -> str:
        """
            Unique string representation of compressor
        """
        return f"({self.stage}, {self.column}, {self.rank})"

    @property
    def instance_name(self) -> str:
        """
            Instance name of compressor
        """
        if self.node_type == CompressorType.FA:
            return f'fa_s{self.stage}_c{self.column}_r{self.rank}'
        elif self.node_type == CompressorType.HA:
            return f'ha_s{self.stage}_c{self.column}_r{self.rank}'
        elif self.node_type == CompressorType.FT:
            return f'ft_s{self.stage}_c{self.column}_r{self.rank}'
        elif self.node_type == CompressorType.PI:
            return f'io_pp_{self.column}[{self.rank}]'
        elif self.node_type == CompressorType.PO:
            if self.rank == 0:
                return f'io_augend[{self.column}]'
            elif self.rank == 1:
                return f'io_addend[{self.column}]'
            else:
                raise RuntimeError(f'Invalid output rank: {self.rank}')
        else:
            raise RuntimeError(f'Invalid node type: {self.node_type}')
        

class PinType(Enum):
    """
        Enum for compressor pin types
    """
    PI   = 1
    A    = 2
    B    = 3
    CI   = 4
    S    = 5
    CO   = 6
    PO   = 7


@dataclass
class Connection:
    """
        Connection data structure, serves as edge in comptree graph view
    """
    src_node: Compressor = None
    dst_node: Compressor = None
    src_pin:  PinType    = None
    dst_pin:  PinType    = None
    delay:    float      = 0  # compatible with legacy mapper

    def set_src_node(self, src_node: Compressor, pin: PinType) -> None:
        """
            Set source node and pin
        """
        self.src_node = src_node
        self.src_pin = pin

    def set_dst_node(self, node: Compressor, pin: PinType) -> None:
        self.dst_node = node
        self.dst_pin = pin

    # derive in which slice the PP should be connected

    @property
    def column(self) -> int:
        assert self.src_node is not None
        assert self.src_pin is not None

        if self.src_pin == PinType.PI:
            return self.src_node.column
        elif self.src_pin == PinType.S:
            return self.src_node.column
        elif self.src_pin == PinType.CO:
            return self.src_node.column + 1
        else:
            raise RuntimeError(f'Invalid source pin type: {self.src_pin}')
        
    @property
    def stage(self) -> int:
        assert self.src_node is not None
        return self.src_node.stage + 1
    
    @property
    def available(self) -> bool:
        return self.dst_node is not None
    
    @property
    def edge(self):
        """
            Unique string representation of connection
        """
        return (str(self.src_node), str(self.dst_node))
    

@dataclass
class CompressorTimer():
    """
        A look-up-table for compressor delay estimation
    """

    fa_delay_a2s:   float = 2
    fa_delay_b2s:   float = 2
    fa_delay_ci2s:  float = 1
    fa_delay_a2co:  float = 3
    fa_delay_b2co:  float = 3
    fa_delay_ci2co: float = 2

    ha_delay_a2s:   float = 1
    ha_delay_b2s:   float = 1
    ha_delay_a2co:  float = 1
    ha_delay_b2co:  float = 1

    def get_fa_delay(self, a: float, b:float, ci: float) -> Tuple[float, float]:
        s = max(
            a + self.fa_delay_a2s,
            b + self.fa_delay_b2s,
            ci + self.fa_delay_ci2s
        )
        co = max(
            a + self.fa_delay_a2co,
            b + self.fa_delay_b2co,
            ci + self.fa_delay_ci2co
        )
        return s, co
    
    def get_ha_delay(self, a: float, b: float) -> Tuple[float, float]:
        s = max(
            a + self.ha_delay_a2s,
            b + self.ha_delay_b2s
        )
        co = max(
            a + self.ha_delay_a2co,
            b + self.ha_delay_b2co
        )
        return s, co
    
    def get_fa_critical_transition(self, a: float, b:float, ci: float) -> Tuple[PinType, PinType]:
        s = np.argmax([
            a + self.fa_delay_a2s,
            b + self.fa_delay_b2s,
            ci + self.fa_delay_ci2s
        ])
        co = np.argmax([
            a + self.fa_delay_a2co,
            b + self.fa_delay_b2co,
            ci + self.fa_delay_ci2co
        ])
        s = [PinType.A, PinType.B, PinType.CI][s]
        co = [PinType.A, PinType.B, PinType.CI][co]
        return s, co
    
    def get_ha_critical_transition(self, a: float, b: float) -> Tuple[PinType, PinType]:
        s = np.argmax([
            a + self.ha_delay_a2s,
            b + self.ha_delay_b2s,
        ])
        co = np.argmax([
            a + self.ha_delay_a2co,
            b + self.ha_delay_b2co,
        ])
        s = [PinType.A, PinType.B][s]
        co = [PinType.A, PinType.B][co]
        return s, co
    

class Nangate45CompressorTimer(CompressorTimer):
    """
        Extracted from Genus + Asap7 typical corner lib
    """

    fa_delay_a2s   = 73.5
    fa_delay_b2s   = 70
    fa_delay_ci2s  = 47.7
    fa_delay_a2co  = 37.3
    fa_delay_b2co  = 33.5
    fa_delay_ci2co = 25.6
    
    ha_delay_a2s   = 41.4
    ha_delay_b2s   = 37.6
    ha_delay_a2co  = 21.8
    ha_delay_b2co  = 20.5


class Asap7CompressorTimer(CompressorTimer):
    """
        Extracted from Genus + Asap7 typical corner lib
        We use middle value collected from 32-bit And-Dadda-Default Multiplier
    """

    fa_delay_a2s   = 29.3166
    fa_delay_b2s   = 29.0568
    fa_delay_ci2s  = 10.4604
    fa_delay_a2co  = 13.5081
    fa_delay_b2co  = 14.4729
    fa_delay_ci2co = 15.5532

    ha_delay_a2s   = 9.5586
    ha_delay_b2s   = 10.4604
    ha_delay_a2co  = 10.2965
    ha_delay_b2co  = 9.5586