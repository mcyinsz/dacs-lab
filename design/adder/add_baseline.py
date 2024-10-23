import numpy as np
from .add_config import PPAdderConfig

def get_Sklansky_adder(input_bit: int) -> PPAdderConfig:
    node_mat = np.zeros((input_bit, input_bit))
    for m in range(input_bit):
        node_mat[m, m] = 1
        tmp = m
        l = m
        d = 1
        while tmp > 0:
            if tmp % 2 == 1:
                l -= d
                node_mat[m, l] = 1
            tmp //= 2
            d *= 2
    return PPAdderConfig(input_bit, required_mat=node_mat)

def get_KoggeStone_adder(input_bit: int) -> PPAdderConfig:
    node_mat = np.zeros((input_bit, input_bit))
    for m in range(input_bit):
        l = m
        d = 1
        while l > 0:
            node_mat[m, l] = 1
            l -= d
            d *= 2
        node_mat[m, 0] = 1
    return PPAdderConfig(input_bit, required_mat=node_mat)

def get_BrentKung_adder(input_bit: int) -> PPAdderConfig:
    node_mat = np.zeros((input_bit, input_bit))
    for m in range(input_bit):
        node_mat[m, m] = 1
        node_mat[m, 0] = 1
    tmp = 2
    while tmp < input_bit:
        for m in range(tmp-1, input_bit, tmp):
            node_mat[m, m-tmp+1] = 1
        tmp *= 2
    return PPAdderConfig(input_bit, required_mat=node_mat)
