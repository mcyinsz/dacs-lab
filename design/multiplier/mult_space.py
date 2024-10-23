
import random
import math
import numpy as np
import time

from .mult_utils import get_minimum_stage
from .pp_generator import PartialProductGenerator
from .comp_estimator import CompressorEstimator
from .comptree_count_view import CompTreeCountView
from .comptree_graph_view import CompTreeGraphView
from .comptree_stage_view import CompTreeStageView
from .mult_baseline import get_dadda_mult_graph, get_wallace_mult_graph

from utils import info

class CompTreeCountViewSpace():

    def __init__(self, m: int, n: int, pp_gen: PartialProductGenerator, comp_est: CompressorEstimator,
                 version='random_v2') -> None:
        self.m = m
        self.n = n
        self.pp_gen = pp_gen
        self.comp_est = comp_est
        self.version = version
        assert m >= 2
        assert n >= 2
        assert self.pp_gen.m == m
        assert self.pp_gen.n == n

    def _get_init_ppcnt_list(self) -> list:
        """
            Return the original partial product count list {a_i}
        """
        return self.pp_gen.get_init_ppcnt_list()

    def _get_minimum_stage(self) -> int:
        """
            Get minimum reduction stage with Dadda multiplier's derivation.
            maximize j s.t. d_j < min(m, n) <= d_{j+1}
            https://en.wikipedia.org/wiki/Dadda_multiplier
        """
        return get_minimum_stage(self.pp_gen.get_num_row())

    def _prune(self, solution: CompTreeCountView) -> bool:
        """
            Prune sub-optimal design points
        """
        s_min = self._get_minimum_stage()
        n_bits = max(self.m, self.n)
        s = solution.num_stage

        return s > s_min + int(math.log(n_bits, 2)) - 2

    def _sample_v1(self, seed: int) -> CompTreeCountView:
        """
            Sample a valid Wallace configuration in matrix representation.
            Guarantees validity, but may not have desired compression stage.
        """
        rng = random.Random(seed)

        while True:
            m, n = self.m, self.n
            ha_list = [0] * (m + n)
            fa_list = [0] * (m + n)
            ppcnt_list = self._get_init_ppcnt_list()

            for i in range(m + n - 1):
                min_fa_cnt = 0
                max_fa_cnt = max(math.ceil((ppcnt_list[i] - 2) / 2), 0)
                fa_list[i] = rng.randint(min_fa_cnt, max_fa_cnt)
                ppcnt_list[i] -= 2 * fa_list[i]
                ppcnt_list[i+1] += fa_list[i]

                min_ha_cnt = max(ppcnt_list[i] - 2, 0)
                max_ha_cnt = max(ppcnt_list[i] - 1, 0)
                ha_list[i] = rng.randint(min_ha_cnt, max_ha_cnt)
                ppcnt_list[i] -= ha_list[i]
                ppcnt_list[i+1] += ha_list[i]

            if ppcnt_list[m + n - 1] == 1 or ppcnt_list[m + n - 1] == 2:
                count_array = np.stack([fa_list, ha_list], axis=1)
                solution = CompTreeCountView(m, n, count_array, self.pp_gen, self.comp_est)

                if not self._prune(solution):
                    return solution
            
            # info('Invalid wallace configuration: {}'.format(ppcnt_list))

    def _sample_v2(self, seed: int) -> CompTreeCountView:
        """
            Sample a valid Wallace configuration in matrix representation.
            Guarantees validity, but may not have desired compression stage.
        """
        rng = random.Random(seed)

        while True:
            m, n = self.m, self.n
            ha_list = [0] * (m + n)
            fa_list = [0] * (m + n)
            ppcnt_list = self._get_init_ppcnt_list()

            for i in range(m + n):
                max_fa_cnt = max(math.ceil((ppcnt_list[i] - 2) / 2), 0)
                min_fa_cnt = max(0, max_fa_cnt - 2)  # use as much fa as possible
                fa_list[i] = rng.randint(min_fa_cnt, max_fa_cnt)
                ppcnt_list[i] -= 2 * fa_list[i]
                if i < m + n - 1:
                    ppcnt_list[i+1] += fa_list[i]

                min_ha_cnt = max(ppcnt_list[i] - 2, 0)
                max_ha_cnt = max(ppcnt_list[i] - 1, 0)
                ha_list[i] = rng.randint(min_ha_cnt, max_ha_cnt)
                ppcnt_list[i] -= ha_list[i]
                if i < m + n - 1:
                    ppcnt_list[i+1] += ha_list[i]

            count_array = np.stack([fa_list, ha_list], axis=1)
            solution = CompTreeCountView(m, n, count_array, self.pp_gen, self.comp_est)

            if not self._prune(solution):
                return solution

            # print(f's = {solution.num_stage}, s_min = {self._get_minimum_stage()}')
            # info('Invalid wallace configuration: {}'.format(ppcnt_list))

    def _sample_manual(self) -> CompTreeCountView:
        """
            Sample count array from manual designs
        """
        rng = random.Random(time.time())
        m, n = self.m, self.n

        select_dadda = rng.random() < 0.5
        if not select_dadda:
            mult_graph = get_wallace_mult_graph(m, n, self.pp_gen)
        else:
            mult_graph = get_dadda_mult_graph(m, n, self.pp_gen)

        graph_view = CompTreeGraphView(m, n, mult_graph, pp_gen=self.pp_gen)
        stage_array = graph_view.to_stage_view()
        stage_view = CompTreeStageView(m, n, stage_array, pp_gen=self.pp_gen)
        count_array = stage_view.to_count_array()
        count_view = CompTreeCountView(m, n, count_array, self.pp_gen, self.comp_est)

        return count_view
        

    def sample(self, seed: int) -> CompTreeCountView:
        if self.version == 'random_v1':
            return self._sample_v1(seed)
        elif self.version == 'random_v2':
            return self._sample_v2(seed)
        elif self.version == 'manual':
            return self._sample_manual()
        else:
            raise ValueError(f'Unknown version: {self.version}')