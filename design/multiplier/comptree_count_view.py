
import math
import numpy as np
from copy import deepcopy

from .pp_generator import PartialProductGenerator, AndPPGenerator
from .comp_estimator import CompressorEstimator, ArchCompEstimator
from .mult_utils import validate_count_array

from utils import assert_error, info, create_hash

class CompTreeCountView():

    def __init__(self, 
                 m: int, 
                 n: int, 
                 count_array: np.ndarray,
                 pp_gen: PartialProductGenerator = None,
                 comp_estimator: CompressorEstimator = None,
    ) -> None:
        self.m = m
        self.n = n
        self.count_array = count_array
        self.pp_gen = pp_gen if pp_gen is not None else AndPPGenerator(m, n)
        self.comp_estimator = comp_estimator if comp_estimator is not None else ArchCompEstimator()

        assert m >= 2
        assert n >= 2
        assert self._validate(self.count_array)

        self._simplify()
        assert self._validate(self.count_array)
        
    # helper properties

    @property
    def full_adder_list(self) -> list:
        return list(self.count_array[:, 0])
    
    @property
    def half_adder_list(self) -> list:
        return list(self.count_array[:, 1])

    @property
    def action_list(self):
        return [
            'add-ha', # add half adder
            'rm-ha',  # remove half adder
            'rp-ha',  # replace half adder to full adder
            'rp-fa'   # replace full adder to half adder
        ]
    
    @property
    def hash(self):
        s = str(tuple(np.reshape(self.count_array, -1)))
        return create_hash(s)

    @property
    def num_stage(self):
        return self._serialize()['num_stage']
    
    @property
    def num_half_adder(self):
        return sum(self.half_adder_list)
    
    @property
    def num_full_adder(self):
        return sum(self.full_adder_list)

    @property
    def area(self):
        fa_area = self.comp_estimator.get_fa_area()
        ha_area = self.comp_estimator.get_ha_area()
        return fa_area * self.num_full_adder + ha_area * self.num_half_adder
    
    # helper functions
    
    def _serialize(self, strict=True) -> dict:
        """
            Serialize the configuration following RL-MUL.
            Specifically, at each stage s, it tries to deplete partial products and prioritize using FA.
            Args:
                strict: if True, use all compressors; if False, compress until every column has 1~2 partial products
            Returns: dict
                seq: a list of tuples, each tuple is (column, is_fa)
                stage: the number of stages
                stage_view: np.ndarray indicating the number of compressors, in shape [#column, #stage, #type]
        """
        m, n = self.m, self.n
        ppcnt_list = self._get_init_ppcnt_list()  # remaining partial products
        ha_list = deepcopy(self.half_adder_list)  # remaining half adders
        fa_list = deepcopy(self.full_adder_list)  # remaining full adders

        stage = 0
        seq = []
        stage_array_list = []

        def condition():
            # Is the compression stage over?
            if strict:
                fa_depleted = all([fa == 0 for fa in fa_list])
                ha_depleted = all([ha == 0 for ha in ha_list])
                return fa_depleted & ha_depleted
            else:
                if ppcnt_list.count(1) + ppcnt_list.count(2) == m + n:
                    return True
                else:
                    return ppcnt_list.count(1) + ppcnt_list.count(2) == m + n - 1 and ppcnt_list[-1] == 0

        while not condition():
            cur_stage_array = np.zeros((m + n, 2), dtype=int)

            for i in reversed(range(m + n)):

                # first, try to use as many FAs as possible
                fa_used = min(
                    math.floor(ppcnt_list[i] / 3),  # limited by remaining PPs
                    fa_list[i]                      # limited by remaining FAs
                )
                fa_list[i] -= fa_used
                ppcnt_list[i] -= 3 * fa_used

                # then, try to use as many HAs as possible
                ha_used = min(
                    math.floor(ppcnt_list[i] / 2),  # limited by remaining PPs
                    ha_list[i]                      # limited by remaining HAs
                )
                ha_list[i] -= ha_used
                ppcnt_list[i] -= 2 * ha_used

                # finally, HAs and FAs carry to higher column
                ppcnt_list[i] += ha_used + fa_used
                if i < m + n - 1:
                    ppcnt_list[i+1] += ha_used + fa_used

                # add HAs and FAs to 
                for _ in range(fa_used):
                    seq.append((i, 1))
                    cur_stage_array[i, 0] += 1
                for _ in range(ha_used):
                    seq.append((i, 0))
                    cur_stage_array[i, 1] += 1
            
            stage += 1
            stage_array_list.append(cur_stage_array)

        strict = False
        assert condition()

        stage_array = np.stack(stage_array_list, axis=2)

        return {
            'seq': seq,
            'num_stage': stage,
            'stage_array': stage_array,
        }

    def _get_init_ppcnt_list(self) -> list:
        return self.pp_gen.get_init_ppcnt_list()
    
    def _get_res_ppcnt_list(self, ha_list: list, fa_list: list) -> list:
        """
            Return the remaining partial product count list
        """
        m, n = self.m, self.n
        res_list = self._get_init_ppcnt_list()

        for i in range(m + n):
            res_list[i] -= ha_list[i] + 2 * fa_list[i]
            if i < m + n - 1:
                res_list[i+1] += ha_list[i] + fa_list[i]
        return res_list

    def _validate(self, count_array: np.ndarray) -> bool:
        """
            Validate the configuration
        """
        return validate_count_array(self.m, self.n, count_array, self._get_init_ppcnt_list())
    
    def _simplify(self) -> None:
        """
            remove redundant compressors to reduce number of stages
        """
        res = self._serialize(strict=False)
        new_count_array = np.zeros((self.m + self.n, 2), dtype=int)

        for col, is_fa in res['seq']:
            comp_type = 0 if is_fa else 1
            new_count_array[col, comp_type] += 1

        self.count_array = new_count_array

    def serialize(self, filepath: str) -> None:
        """
            Write the serialzation to a file
        """
        m, n = self.m, self.n
        compressor_cnt = sum(self.half_adder_list) + sum(self.full_adder_list)
        res = self._serialize()

        with open(filepath, 'w') as f:
            f.write('%d %d\n' % (m, n))
            f.write('%d\n' % (compressor_cnt))
            f.write('\n'.join([" ".join([str(i) for i in t]) for t in res['seq']]))
            f.write('\n')

        info('WallaceConfig: m=%d, n=%d' % (m, n))
        info('Half adder list: %s' % self.half_adder_list)
        info('Full adder list: %s' % self.full_adder_list)
        info('Compressor count: %d' % compressor_cnt)
        info('Stage: %d' % res['stage'])
    
    def mutate(self, column: int, action: str) -> np.ndarray:
        """
            Mutate the configuration and legalize, following RL-MUL
            Args:
                column (int): the column index to mutate
                action (str): the action to take
            Returns:
                np.ndarry: new config if the mutation is successful,
                None: if the mutation is not successful 
        """

        m, n = self.m, self.n
        ha_list = deepcopy(self.half_adder_list)
        fa_list = deepcopy(self.full_adder_list)

        # remaining partial product list
        res_list = self._get_res_ppcnt_list(ha_list, fa_list)

        # define the actions, which adjust ha/fa/res_list accordingly
        def add_half_adder(col: int):
            ha_list[col] += 1
            res_list[col] -= 1
            if col < m + n - 1:
                res_list[col+1] += 1

        def add_full_adder(col):
            fa_list[col] += 1
            res_list[col] -= 2
            if col < m + n - 1:
                res_list[col+1] += 1

        def remove_half_adder(col: int):
            assert ha_list[col] > 0
            ha_list[col] -= 1
            res_list[col] += 1
            if col < m + n - 1:
                res_list[col+1] -= 1

        def remove_full_adder(col: int):
            assert fa_list[col] > 0
            fa_list[col] -= 1
            res_list[col] += 2
            if col < m + n - 1:
                res_list[col+1] -= 1

        def replace_half_adder(col: int):
            assert ha_list[col] > 0
            ha_list[col] -= 1
            fa_list[col] += 1
            res_list[col] -= 1

        def replace_full_adder(col: int):
            assert fa_list[col] > 0
            fa_list[col] -= 1
            ha_list[col] += 1
            res_list[col] += 1

        # validate and apply compressor adjustment 
        if action == self.action_list[0]:   # add half adder
            if res_list[column] != 2:
                return None
            add_half_adder(column)
        elif action == self.action_list[1]: # remove half adder
            if res_list[column] != 1 or ha_list[column] == 0:
                return None
            remove_half_adder(column)
        elif action == self.action_list[2]: # replace half adder to full adder
            if res_list[column] != 2 or ha_list[column] == 0:
                return None
            replace_half_adder(column)
        elif action == self.action_list[3]: # replace full adder to half adder
            if res_list[column] != 1 or fa_list[column] == 0:
                return None
            replace_full_adder(column)
        else:
            assert_error('Invalid action %s' % action)

        # update res list and legalize
        for i in range(column + 1, m + n):
            res = res_list[i]
            if res == 1 or res == 2:
                continue
            elif res == 3:
                if ha_list[i] > 0:
                    replace_half_adder(i)
                else:
                    add_full_adder(i)
            elif res == 0:
                if ha_list[i] > 0:
                    remove_half_adder(i)
                elif fa_list[i] > 0:
                    remove_full_adder(i)
                else:
                    assert_error('Column %d has no remaining PP or compressor' % (i))
            else:
                assert_error('Invalid res %d at column %d' % (res, i))

        new_count_array = np.stack([fa_list, ha_list], axis=1)
        if self._validate(new_count_array):
            return new_count_array
        else:
            return None

    def to_stage_array(self) -> np.ndarray:
        """
            Convert the configuration to stage array
        """
        return self._serialize()['stage_array']