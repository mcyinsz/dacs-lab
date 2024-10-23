import abc
from typing import Tuple
import numpy as np

from .mult_utils import MultGraphNode, MultGraphEdge

class CompressorEstimator(abc.ABC):

    @abc.abstractmethod
    def get_fa_delay(self, a: float, b:float, ci: float) -> Tuple[float, float]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_ha_delay(self, a: float, b: float) -> Tuple[float, float]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_fa_area(self) -> float:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_ha_area(self) -> float:
        raise NotImplementedError()
    

class ArchCompEstimator(CompressorEstimator):

    def __init__(
        self,
        use_maj: bool = False,
        use_xor3: bool = False,
        use_or3: bool = False,
    ) -> None:
        super().__init__()
        self.use_maj = use_maj
        self.use_xor3 = use_xor3
        self.use_or3 = use_or3

    def get_fa_delay(self, a: float, b:float, ci: float) -> Tuple[float, float]:
        if self.use_xor3:
            s = max(a, b, ci) + 1             # s = a ^ b ^ ci, 1 XOR3 delay
        else:
            s = max(max(a, b) + 1, ci) + 1    # s = (a ^ b) ^ ci, 2 XOR2 delay

        if self.use_maj:
            co = max(a, b, ci) + 1            # co = MAJ3(a, b, ci), 1 MAJ3 delay
        elif self.use_or3:
            co = max(a, b, ci) + 2            # co = OR3(a & b, a & ci, b & ci), 1 AND2 delay + 1 OR3 delay
        else:
            # HA1.carry = a & b, 1 AND delay
            # HA1.sum   = a ^ b, 1 XOR delay
            # HA2.carry = HA1.sum & ci, 1 XOR delay + 1 AND delay
            # FA.carry = HA1.carry | HA2.carry, 1 XOR delay + 1 AND delay + 1 OR delay
            co = max(max(a, b) + 1, ci) + 2
        return s, co
    
    def get_ha_delay(self, a: float, b: float) -> Tuple[float, float]:
        s = max(a, b) + 1    # s = a ^ b, 1 XOR delay
        co = max(a, b) + 1   # co = a & b, 1 AND delay
        return s, co
    
    def get_fa_area(self) -> float:
        return 3  # this comes from GOMIL and is definitely wrong for asap7
    
    def get_ha_area(self) -> float:
        return 2


class SynthCompEstimator(CompressorEstimator):

    def __init__(self) -> None:
        super().__init__()

        self.fa_delay_a2s   = 0
        self.fa_delay_b2s   = 0
        self.fa_delay_ci2s  = 0
        self.fa_delay_a2co  = 0
        self.fa_delay_b2co  = 0
        self.fa_delay_ci2co = 0

        self.ha_delay_a2s   = 0
        self.ha_delay_b2s   = 0
        self.ha_delay_a2co  = 0
        self.ha_delay_b2co  = 0
        
        self.fa_area = 0
        self.ha_area = 0

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
    
    def get_fa_area(self) -> float:
        return self.fa_area

    def get_ha_area(self) -> float:
        return self.ha_area
    
    def get_fa_critical_transition(self, a: float, b:float, ci: float) -> Tuple[int, int]:
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
        s = [MultGraphEdge.PIN_TYPE_COMP_A, MultGraphEdge.PIN_TYPE_COMP_B, MultGraphEdge.PIN_TYPE_COMP_CI][s]
        co = [MultGraphEdge.PIN_TYPE_COMP_A, MultGraphEdge.PIN_TYPE_COMP_B, MultGraphEdge.PIN_TYPE_COMP_CI][co]
        return s, co
    
    def get_ha_critical_transition(self, a: float, b: float) -> Tuple[int, int]:
        s = np.argmax([
            a + self.ha_delay_a2s,
            b + self.ha_delay_b2s,
        ])
        co = np.argmax([
            a + self.ha_delay_a2co,
            b + self.ha_delay_b2co,
        ])
        s = [MultGraphEdge.PIN_TYPE_COMP_A, MultGraphEdge.PIN_TYPE_COMP_B][s]
        co = [MultGraphEdge.PIN_TYPE_COMP_A, MultGraphEdge.PIN_TYPE_COMP_B][co]
        return s, co


class Asap7CompEstimator(SynthCompEstimator):
    """
        Extracted from Genus + Asap7 typical corner lib
        We use middle value collected from 32-bit And-Dadda-Default Multiplier
    """

    def __init__(self) -> None:
        super().__init__()

        self.fa_delay_a2s   = 29.3166
        self.fa_delay_b2s   = 29.0568
        self.fa_delay_ci2s  = 10.4604
        self.fa_delay_a2co  = 13.5081
        self.fa_delay_b2co  = 14.4729
        self.fa_delay_ci2co = 15.5532

        self.ha_delay_a2s   = 9.5586
        self.ha_delay_b2s   = 10.4604
        self.ha_delay_a2co  = 10.2965
        self.ha_delay_b2co  = 9.5586

        self.fa_area = 6.998
        self.ha_area = 3.499

class Nangate45CompEstimator(SynthCompEstimator):
    """
        Extracted from Genus + Asap7 typical corner lib
    """

    def __init__(self) -> None:

        self.fa_delay_a2s   = 73.5
        self.fa_delay_b2s   = 70
        self.fa_delay_ci2s  = 47.7
        self.fa_delay_a2co  = 37.3
        self.fa_delay_b2co  = 33.5
        self.fa_delay_ci2co = 25.6

        self.ha_delay_a2s   = 41.4
        self.ha_delay_b2s   = 37.6
        self.ha_delay_a2co  = 21.8
        self.ha_delay_b2co  = 20.5

        self.fa_area = 10.374
        self.ha_area = 5.852


# some pre-defined compressor estimators

NANGATE45_ARCH_COMP_ESTIMATOR = ArchCompEstimator(use_maj=False, use_xor3=False, use_or3=False)

ASAP7_ARCH_COMP_ESTIMATOR = ArchCompEstimator(use_maj=True, use_xor3=False, use_or3=False)

NANGATE45_SYNTH_COMP_ESTIMATOR = Nangate45CompEstimator()

ASAP7_SYNTH_COMP_ESTIMATOR = Asap7CompEstimator()