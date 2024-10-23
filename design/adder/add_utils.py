import numpy as np
from .add_config import PPAdderConfig
from .add_space import *
import matplotlib.pyplot as plt
import os
from utils import mkdir

def pareto_front(data):
    is_pareto = np.ones(data.shape[0], dtype=bool)
    for i, c in enumerate(data):
        is_pareto[i] = np.all(np.any(data[:i] > c, axis=1)) and np.all(np.any(data[i+1:] > c, axis=1))
    return data[is_pareto]

def plot_distribution(adders_list, save_path, n_bit, infos=None, plot_pareto=False):
    for i, adders in enumerate(adders_list):
        features = [(c.level, c.area) for c in adders]

        if infos != None:
            info = infos[i]
        else:
            info = {"label": f'data (num={len(adders)})', "color": f"C{i}"}
        plt.scatter(*zip(*features), **info)

        if plot_pareto:
            pareto_fronts = pareto_front(np.array(features))
            pareto_fronts = pareto_fronts[np.argsort(pareto_fronts[:, 0]), :]
            plt.plot(pareto_fronts[:, 0], pareto_fronts[:, 1], label='Pareto Front', color=info["color"])


    # plot classical sklansky design
    classical_sklansky_config = get_Sklansky_adder(input_bit=n_bit)
    plt.scatter(classical_sklansky_config.level, classical_sklansky_config.area, color='C2', label='Sklansky')

    # plot classical brentkung design
    classical_brentkung_config = get_BrentKung_adder(input_bit=n_bit)
    plt.scatter(classical_brentkung_config.level, classical_brentkung_config.area, color='C4', label='BrentKung')

    plt.title(f'PPAdder data distribution bit={n_bit}')
    plt.xlabel('Depth')
    plt.ylabel('Area')
    plt.legend()
    mkdir(save_path)
    plt.savefig(os.path.join(save_path, f'{n_bit}-distribution.png'))

def plot_distribution_black_box(results_list, save_path, n_bit, infos=None):
    for i, results in enumerate(results_list):

        if infos != None:
            info = infos[i]
        else:
            info = {"label": f'data (num={len(results)})', "color": f"C{i}"}
        plt.scatter(*zip(*results), **info)

    plt.title(f'PPAdder data distribution bit={n_bit}')
    plt.xlabel('Timing')
    plt.ylabel('Area')
    plt.legend()
    mkdir(save_path)
    plt.savefig(os.path.join(save_path, f'{n_bit}-blackbox-distribution.png'))

def generate_adder_default_verilog(n_bit) -> str:
    codes = """
module PPAdder (
    input wire clk,
    input [%d:0] io_augend,
    input [%d:0] io_addend,
    output [%d:0] io_outs
);
    assign io_outs = io_augend + io_addend;
endmodule
"""% (n_bit-1, n_bit-1, n_bit-1)
    return codes
