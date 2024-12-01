import abc
import os
import subprocess

from design.multiplier.pp_generator import PartialProductGenerator
from design.multiplier.comp_tree import CompTreeGraphView
from design.adder.add_config import AdderConfig
from tempfile import TemporaryDirectory

from utils import mkdir, execute, remove, if_exist

class CompTreeTopModule(abc.ABC):
    """
        Top module for compressor tree-based datapath modules,
        which typically consists PPG, CT and CPA submodules
    """

    def __init__(self, comp_tree: CompTreeGraphView, final_adder: AdderConfig) -> None:
        self.comp_tree = comp_tree
        self.final_adder = final_adder
        self.pp_gen = comp_tree.pp_gen
        assert comp_tree.num_column == final_adder.num_bit

    @abc.abstractmethod
    def generate_verilog(
        self,
        top_module_name: str = 'CompTreeTopModule',
        ppg_name: str = 'PartialProductGenerator',
        ct_name: str = 'CompressorTree',
        cpa_name: str = 'CarryPropagateAdder',
    ) -> str:
        """
            Generate verilog code for the top module
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def generate_testbench(
        self,
        top_module_name: str = 'CompTreeTopModule',
        testbench_name: str = 'CompTreeTopModuleTestbench',
        n_testcase: int = 100,
        random_seed: int = 42,
    ) -> str:
        raise NotImplementedError
    
    
    def run_testbench(
        self,
        rundir: str = None,
    ) -> bool:
        """
            Use iverilog to simulate the testbench, return if the simulation is successful
        """
        mkdir(rundir)
        verilog_path = os.path.join(rundir, 'top.v')
        testbench_path = os.path.join(rundir, 'testbench.v')

        with open(verilog_path, 'w') as f:
            f.write(self.generate_verilog())

        with open(testbench_path, 'w') as f:
            f.write(self.generate_testbench())

        cmd = "cd %s && iverilog -o sim.vvp top.v testbench.v && vvp sim.vvp | tee sim.log" % rundir
        ret = subprocess.run(["/bin/bash", "-c", cmd])
        if not if_exist(os.path.join(rundir, 'sim.log')):
            return False

        cmd = "cat sim.log | grep 'FATAL' | wc -l"
        ret = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True)
        num_fatal = int(ret.stdout.strip())

        return num_fatal == 0