import numpy as np

from .count_view import CompTreeCountView
from design.multiplier.pp_generator import PartialProductGenerator
from design.multiplier.mult_utils import InvalidStageViewError, InvalidCountViewError

from utils import assert_error, info, create_hash, debug

class CompTreeStageView():
    """
        Stage view of compressor tree
        The core of stage view is a 2darray of shape [L, 2, S], 
        where L is the number of columns, S is the number of stages
        Each elements represents the number of full adders or half adders in a specific column.
    """

    def __init__(self,
                stage_array: np.ndarray,
                pp_gen: PartialProductGenerator
    ) -> None:
        self.stage_array = stage_array
        self.pp_gen = pp_gen
        if not self.validate(stage_array):
            raise InvalidStageViewError(f'Invalid stage array: {stage_array}')

    # helper properties

    @property
    def num_column(self):
        return self.stage_array.shape[0]

    @property
    def num_stage(self):
        return self.stage_array.shape[2]    
    
    @property
    def hash(self):
        s = str(tuple(np.reshape(self.stage_array, -1)))
        return create_hash(s)

    # helper functions

    def get_res_ppcnt_array(self, stage_array: np.ndarray = None) -> np.ndarray:
        """
            Return 2d-array of the remaining partial product after every column & stage
        """
        if stage_array is None:
            stage_array = self.stage_array

        ppcnt_vec = np.array(self.pp_gen.get_init_ppcnt_list())
        res_ppcnt_array = np.zeros((self.num_column, self.num_stage), dtype=np.int64)

        for s in range(self.num_stage):
            fa_vec = stage_array[:, 0, s]
            ha_vec = stage_array[:, 1, s]

            ppcnt_vec -= (3 * fa_vec + 2 * ha_vec)
            res_ppcnt_array[:, s] = ppcnt_vec

            sum_vec = fa_vec + ha_vec
            carry_vec = np.concatenate([np.zeros(1, dtype=np.int64), sum_vec[:-1]])
            ppcnt_vec += (sum_vec + carry_vec)

        return res_ppcnt_array


    def validate(self, stage_array: np.ndarray) -> bool:
        """
            Validate the configuration
        """
        if not stage_array.shape[0] == self.pp_gen.num_column:
            return False
        if not stage_array.shape[1] == 2:
            return False
        if not np.all(stage_array >= 0):
            return False
        
        # check over-compression
        res_ppcnt_array = self.get_res_ppcnt_array(stage_array)
        if not np.all(res_ppcnt_array >= 0):
            return False

        # check under-compression
        try:
            count_array = np.sum(stage_array, axis=2)
            count_view = CompTreeCountView(count_array, self.pp_gen)
        except InvalidCountViewError:
            return False
        
        return True

    
    def to_count_view(self) -> CompTreeCountView:
        count_array = np.sum(self.stage_array, axis=2)
        return CompTreeCountView(count_array, self.pp_gen)
    

    def mutate(self, column: int, src_stage: int, dst_stage: int) -> np.ndarray:
        """
            Swap FA in src stage and HA in dst stage
            Returns:
                np.ndarry: new config if the mutation is successful,
                None: if the mutation is not successful 
        """

        new_stage_array = self.stage_array.copy()
        if new_stage_array[column, 0, src_stage] == 0: return None
        if new_stage_array[column, 1, dst_stage] == 0: return None
        new_stage_array[column, 0, src_stage] -= 1
        new_stage_array[column, 1, src_stage] += 1
        new_stage_array[column, 0, dst_stage] += 1
        new_stage_array[column, 1, dst_stage] -= 1

        if self.validate(new_stage_array):
            return new_stage_array
        else:
            return None