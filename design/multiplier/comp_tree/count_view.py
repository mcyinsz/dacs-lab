
import numpy as np
from copy import deepcopy

from design.multiplier.pp_generator import PartialProductGenerator
from design.multiplier.mult_utils import InvalidCountViewError

from utils import assert_error, info, create_hash


class CompTreeCountView():
    """
        Count view of compressor tree
        The core of count view is a 2darray of shape [L, 2], where L is the number of columns.
        Each elements represents the number of full adders or half adders in a specific column.
    """

    def __init__(self, 
                 count_array: np.ndarray,
                 pp_gen: PartialProductGenerator,
    ) -> None:
        self.count_array = count_array
        self.pp_gen = pp_gen
        if not self.validate(count_array):
            raise InvalidCountViewError(f'Invalid count array: {count_array}')
        

    # helper properties

    @property
    def full_adder_list(self) -> list:
        return list(self.count_array[:, 0])
    
    @property
    def half_adder_list(self) -> list:
        return list(self.count_array[:, 1])
    
    @property
    def num_column(self) -> int:
        return self.count_array.shape[0]
    
    @property
    def hash(self):
        s = str(tuple(np.reshape(self.count_array, -1)))
        return create_hash(s)
    
    def get_init_ppcnt_list(self) -> list:
        return self.pp_gen.get_init_ppcnt_list()
    
    def get_res_ppcnt_list(self) -> list:
        """
            Return the remaining partial product count list
        """
        ha_list = self.half_adder_list
        fa_list = self.full_adder_list
        res_list = self.get_init_ppcnt_list()
        num_columns = len(res_list)

        for i in range(num_columns):
            res_list[i] -= ha_list[i] + 2 * fa_list[i]
            if i < num_columns - 1:
                res_list[i+1] += ha_list[i] + fa_list[i]
        return res_list

    def validate(self, count_array: np.ndarray) -> bool:
        """
            Validate the configuration
        """
        if not count_array.shape[0] == self.pp_gen.num_column:
            return False
        if not count_array.shape[1] == 2:
            return False
        if not np.all(count_array >= 0):
            return False

        res_ppcnt_array = np.array(self.get_res_ppcnt_list())

        eq_one = np.equal(res_ppcnt_array, 1)
        eq_two = np.equal(res_ppcnt_array, 2)
        eq_zero = np.equal(res_ppcnt_array, 0)
        appending_zeros = np.cumprod(eq_zero[::-1])[::-1]
        # as a special case (e.g. Dadda multiplier), we allow the presence of leading zero sequence

        if np.all(eq_one + eq_two + appending_zeros):
            return True
        else:
            return False


    # mutate functions

    @property
    def action_list(self):
        return [
            'add-ha', # add half adder
            'rm-ha',  # remove half adder
            'rp-ha',  # replace half adder to full adder
            'rp-fa'   # replace full adder to half adder
        ]
    
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

        num_column = self.num_column
        ha_list = deepcopy(self.half_adder_list)
        fa_list = deepcopy(self.full_adder_list)

        # remaining partial product list
        res_list = self.get_res_ppcnt_list()

        # define the actions, which adjust ha/fa/res_list accordingly
        def add_half_adder(col: int):
            ha_list[col] += 1
            res_list[col] -= 1
            if col < num_column - 1:
                res_list[col+1] += 1

        def add_full_adder(col):
            fa_list[col] += 1
            res_list[col] -= 2
            if col < num_column - 1:
                res_list[col+1] += 1

        def remove_half_adder(col: int):
            assert ha_list[col] > 0
            ha_list[col] -= 1
            res_list[col] += 1
            if col < num_column - 1:
                res_list[col+1] -= 1

        def remove_full_adder(col: int):
            assert fa_list[col] > 0
            fa_list[col] -= 1
            res_list[col] += 2
            if col < num_column - 1:
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
        for i in range(column + 1, num_column):
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
        if self.validate(new_count_array):
            return new_count_array
        else:
            return None