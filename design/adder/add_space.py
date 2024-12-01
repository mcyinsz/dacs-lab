import numpy as np
from .add_config import PPAdderConfig
from .add_baseline import *

class PPAdderSpace():

    def __init__(self, input_bit) -> None:
        self.N = input_bit
    
    def sklansky_sample(self) -> PPAdderConfig:
        return get_Sklansky_adder(self.N)
    
    def koggestone_sample(self) -> PPAdderConfig:
        return get_KoggeStone_adder(self.N)
    
    def brentkung_sample(self) -> PPAdderConfig:
        return get_BrentKung_adder(self.N)
    
    def _sample(self, seed: int) -> PPAdderConfig:
        np.random.seed(seed)
        length = int((self.N - 1) * (self.N - 2) / 2)
        required_vector = np.random.randint(2, size=length)
        return PPAdderConfig(input_bit=self.N, required_vector=required_vector)

    def sample(self, seed: int) -> PPAdderConfig:
        return self._sample(seed)
