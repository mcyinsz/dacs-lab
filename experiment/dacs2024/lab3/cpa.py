from env import *

import random

from design.adder.add_config import PPAdderConfig
from design.adder.add_baseline import get_Sklansky_adder, get_KoggeStone_adder, get_BrentKung_adder


def get_default_cpa(adder_type='sklansky', input_bit=N_BIT*2) -> PPAdderConfig:
    """
        Get a default carry-propagate adder
    """

    if adder_type == 'sklansky':
        adder = get_Sklansky_adder(input_bit)
    elif adder_type == 'koggestone':
        adder = get_KoggeStone_adder(input_bit)
    elif adder_type == 'brentkung':
        adder = get_BrentKung_adder(input_bit)
    else:  # fall back to ripple carry adder
        adder = PPAdderConfig(input_bit)

    return adder


def get_cpa(input_bit=N_BIT*2, iter=0, seed=42) -> PPAdderConfig:
    """
        An exmaple code to modify the adder structrue with random action
        Reference PrefixRL: https://ieeexplore.ieee.org/abstract/document/9586094/
    """
    adder = get_default_cpa(input_bit=input_bit)

    rng = random.Random(seed)
    flag = False  # if the action is effective

    for i in range(iter):
        msb = random.randint(0, input_bit-1)
        lsb = random.randint(0, msb)
        
        if rng.random() < 0.5:
            flag = adder.add(msb, lsb)
        else:
            flag = adder.delete(msb, lsb)

    return adder