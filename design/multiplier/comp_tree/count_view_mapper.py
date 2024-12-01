import abc
import math
import numpy as np
from copy import deepcopy

from .count_view import CompTreeCountView
from .stage_view import CompTreeStageView

class CompTreeCountViewMapper(abc.ABC):
    """
        Abstract class for mapping compressor tree count view to stage view
    """

    @abc.abstractmethod
    def _map(self, count_view: CompTreeCountView) -> CompTreeStageView:
        """
            Map compressor tree count view to stage view
        """
        raise NotImplementedError

    def __call__(self, count_view: CompTreeCountView) -> CompTreeStageView:
        """
            Map compressor tree count view to stage view
        """
        return self._map(count_view)
    

class SerialCompTreeCountViewMapper(CompTreeCountViewMapper):
    """
        Mapper for serializing compressor tree count view to stage view
    """

    def _serialize(self, count_view: CompTreeCountView, strict=True) -> dict:
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
        num_column = count_view.num_column
        ppcnt_list = count_view.get_init_ppcnt_list()   # remaining partial products
        ha_list = deepcopy(count_view.half_adder_list)  # remaining half adders
        fa_list = deepcopy(count_view.full_adder_list)  # remaining full adders

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
                res_ppcnt_array = np.array(ppcnt_list)
                eq_one = np.equal(res_ppcnt_array, 1)
                eq_two = np.equal(res_ppcnt_array, 2)
                eq_zero = np.equal(res_ppcnt_array, 0)
                appending_zeros = np.cumprod(eq_zero[::-1])[::-1]

                if np.all(eq_one + eq_two + appending_zeros):
                    return True
                else:
                    return False

        while not condition():
            cur_stage_array = np.zeros((num_column, 2), dtype=int)

            for i in reversed(range(num_column)):

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
                if i < num_column - 1:
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

    def simplify(self, count_view: CompTreeCountView) -> CompTreeCountView:
        """
            remove redundant compressors to reduce number of stages
        """
        res = self._serialize(strict=False)
        num_column = count_view.num_column
        new_count_array = np.zeros((num_column, 2), dtype=int)

        for col, is_fa in res['seq']:
            comp_type = 0 if is_fa else 1
            new_count_array[col, comp_type] += 1

        return CompTreeCountView(new_count_array, count_view.pp_gen)

    def _map(self, count_view: CompTreeCountView) -> CompTreeStageView:
        """
            Map compressor tree count view to stage view
        """
        res = self._serialize(count_view)
        return CompTreeStageView(res['stage_array'], count_view.pp_gen)