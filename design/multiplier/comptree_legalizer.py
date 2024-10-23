import os
import numpy as np

from .pp_generator import PartialProductGenerator
from .mult_utils import LegalizationError, InvalidActionError

class CompTreeStageViewLegalizer():

    def __init__(
        self,
        stage_array: np.ndarray,
        pp_gen: PartialProductGenerator = None,
    ):
        assert len(stage_array.shape) == 3
        assert stage_array.shape[1] == 2
        self.n_column = stage_array.shape[0]
        self.n_bit = stage_array.shape[0] // 2
        self.n_stage = stage_array.shape[2]
        self.stage_array = stage_array
        self.pp_gen = pp_gen if pp_gen is not None else AndPPGenerator(self.n_bit, self.n_bit)        

        # init pp at every stage
        init_pp_array = np.zeros((self.n_column, self.n_stage + 1), dtype=np.int64)
        init_pp_array[:, 0] = np.array(self.pp_gen.get_init_ppcnt_list())

        for i in range(1, self.n_stage + 1):
            fa_array = stage_array[:, 0, i-1]
            ha_array = stage_array[:, 1, i-1]
            sum_array = fa_array + ha_array
            carry_array = np.concatenate([np.zeros(1, dtype=np.int64), sum_array[:-1]])
            init_pp_array[:, i:i+1] = np.reshape(init_pp_array[:, i-1] - (3 * fa_array + 2 * ha_array) + (sum_array + carry_array), (-1, 1))

        self.init_pp_array = init_pp_array

    def get_res_pp(self, column, stage, init_pp=None):
        """
            Get residual pp at given stage
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage
        if init_pp is None:
            init_pp = self.init_pp_array[column, stage]
        fa = self.stage_array[column, 0, stage]
        ha = self.stage_array[column, 1, stage]
        return int(init_pp - 3 * fa - 2 * ha)
    
    def check_position(self, column, stage, init_pp=None) -> bool:
        """
            Check whether the position is valid (True)
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage <= self.n_stage

        if init_pp is None:
            init_pp = self.init_pp_array[column, stage]
        
        if stage == self.n_stage:
            if column == self.n_column - 1:
                return init_pp in (0, 1, 2)
            else:
                return init_pp in (1, 2)
        else:
            res_pp = self.get_res_pp(column, stage, init_pp=init_pp)
            return res_pp >= 0
        
    
    def adjust_pp(self, column, stage, sum_delta, carry_delta, modify=False) -> list:
        """
            Adjust pp after chaning compressors, and report potential invalid assignment (only in other stages!)
            This function cannot handle illegal cases where res pp < 0 at given stage
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage

        invalid_list = []

        for s in range(stage + 1, self.n_stage + 1):
            # adjust current column
            init_pp = self.init_pp_array[column, s] + sum_delta
            if not self.check_position(column, s, init_pp=init_pp):
                invalid_list.append((column, s))
            if modify:
                self.init_pp_array[column, s] = init_pp

            # adjust next column
            if column == self.n_column - 1:
                continue
            init_pp = self.init_pp_array[column + 1, s] + carry_delta
            if not self.check_position(column + 1, s, init_pp=init_pp):
                invalid_list.append((column + 1, s))
            if modify:
                self.init_pp_array[column + 1, s] = init_pp

        return invalid_list

    def replace_fa(self, column, stage, modify=False) -> list:
        """
            Replace FA to HA
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage

        if self.stage_array[column, 0, stage] <= 0:
            raise InvalidActionError(f'Not enough FA at column={column}, stage={stage}')

        # result of adjust pp
        invalid_list = []
        if self.get_res_pp(column, stage) + 1 < 0:
            invalid_list.append((column, stage))
        invalid_list += self.adjust_pp(column, stage, sum_delta=1, carry_delta=0, modify=modify)

        # adjust compressor
        if modify:
            self.stage_array[column:column+1, 0:1, stage:stage+1] -= 1
            self.stage_array[column:column+1, 1:2, stage:stage+1] += 1

        return invalid_list
    
    def replace_ha(self, column, stage, modify=False) -> list:
        """
            Replace HA to FA
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage

        if self.stage_array[column, 1, stage] <= 0:
            raise InvalidActionError(f'Not enough HA at column={column}, stage={stage}')

        # result of adjust pp
        invalid_list = []
        if self.get_res_pp(column, stage) - 1 < 0:
            invalid_list.append((column, stage))
        invalid_list += self.adjust_pp(column, stage, sum_delta=-1, carry_delta=0, modify=modify)

        # adjust compressor
        if modify:
            self.stage_array[column:column+1, 1:2, stage:stage+1] -= 1
            self.stage_array[column:column+1, 0:1, stage:stage+1] += 1

        return invalid_list
    
    def split_fa(self, column, stage, modify=False) -> list:
        """
            Split FA to HA
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage

        if self.stage_array[column, 0, stage] <= 0:
            raise InvalidActionError(f'Not enough FA at column={column}, stage={stage}')

        # result of adjust pp
        invalid_list = []
        if self.get_res_pp(column, stage) - 1 < 0:
            invalid_list.append((column, stage))
        invalid_list += self.adjust_pp(column, stage, sum_delta=0, carry_delta=1, modify=modify)

        # adjust compressor
        if modify:
            self.stage_array[column:column+1, 0:1, stage:stage+1] -= 1
            self.stage_array[column:column+1, 1:2, stage:stage+1] += 2

        return invalid_list
    
    def fuse_ha(self, column, stage, modify=False) -> list:
        """
            Fuse 2xHA to FA
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage

        if self.stage_array[column, 1, stage] <= 1:
            raise InvalidActionError(f'Not enough HA at column={column}, stage={stage}')

        # result of adjust pp
        invalid_list = []
        if self.get_res_pp(column, stage) + 1 < 0:
            invalid_list.append((column, stage))
        invalid_list += self.adjust_pp(column, stage, sum_delta=0, carry_delta=-1, modify=modify)

        # adjust compressor
        if modify:
            self.stage_array[column:column+1, 1:2, stage:stage+1] -= 2
            self.stage_array[column:column+1, 0:1, stage:stage+1] += 1

        return invalid_list

    def delete_ha(self, column, stage, modify=False) -> list:
        """
            Delete HA
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage

        if self.stage_array[column, 1, stage] <= 0:
            raise InvalidActionError(f'Not enough HA at column={column}, stage={stage}')

        # result of adjust pp
        invalid_list = []
        if self.get_res_pp(column, stage) + 2 < 0:
            invalid_list.append((column, stage))
        invalid_list += self.adjust_pp(column, stage, sum_delta=1, carry_delta=-1, modify=modify)

        # adjust compressor
        if modify:
            self.stage_array[column:column+1, 1:2, stage:stage+1] -= 1

        return invalid_list
    
    def add_ha(self, column, stage, modify=False) -> list:
        """
            Add HA
        """
        assert column >= 0 and column < self.n_column
        assert stage >= 0 and stage < self.n_stage

        # result of adjust pp
        invalid_list = []
        if self.get_res_pp(column, stage) - 2 < 0:
            invalid_list.append((column, stage))
        invalid_list += self.adjust_pp(column, stage, sum_delta=-1, carry_delta=1, modify=modify)

        # adjust compressor
        if modify:
            self.stage_array[column:column+1, 1:2, stage:stage+1] += 1

        return invalid_list
    
    
    def __call__(self, max_trial=1000, max_repeat=5) -> np.ndarray:
        """
            Legalize stage array, return the legalized array.
            Throw LegalizationFailure if it fails
        """

        invalid_queue = []

        # initialize invalid queue
        for s in range(self.n_stage + 1):
            for c in range(self.n_column):
                if not self.check_position(c, s):
                    invalid_queue.append((c, s))
        
        # start to legalize
        n_trial = 0
        action_cnt_dict = dict()

        while len(invalid_queue) > 0:
            column, stage = invalid_queue.pop(0)
            action_list = []
            action_idx = 0

            # double check whether the problem has been fixed
            if self.check_position(column, stage):
                continue
            # print(f'[Trial {n_trial}] Problem: column={column}, stage={stage}')

            def trial_action(action, c, s):
                nonlocal action_list
                nonlocal action_idx
                try:
                    invalid_list = action(c, s)
                    action_list.append((len(invalid_list), action_idx, action, c, s))
                    action_idx += 1
                except InvalidActionError:
                    return

            # res pp < 0
            if stage < self.n_stage:
                
                # replace FA with HA at column, anytime
                for s in reversed(range(stage + 1)):
                    trial_action(self.replace_fa, column, s)

                # split FA at column - 1, ASAP
                if column > 0:
                    for s in range(stage):
                        trial_action(self.split_fa, column - 1, s)

                # delete HA at column
                trial_action(self.delete_ha, column, stage)

            # final pp < 1
            elif self.init_pp_array[column, stage] < 1:

                # replace FA with HA at column, anytime
                for s in reversed(range(stage)):
                    trial_action(self.replace_fa, column, s)

                # split FA at column - 1, ASAP
                if column > 0:
                    for s in range(stage):
                        trial_action(self.split_fa, column - 1, s)

                # delete HA at column, ASAP
                for s in range(stage):
                    trial_action(self.delete_ha, column, s)

            # final pp > 2
            else:

                # replace HA with FA at column, anytime
                for s in reversed(range(stage)):
                    trial_action(self.replace_ha, column, s)

                # fuse 2x HA at column - 1, ASAP
                if column > 0:
                    for s in range(stage):
                        trial_action(self.fuse_ha, column - 1, s)

                # add HA at column, ASAP
                for s in range(stage):
                    trial_action(self.add_ha, column, s)

            # greedily select for action with minimum incurred invalid assignment
            action_list = sorted(action_list, key=lambda x: (x[0], x[1]))

            for _, _, final_action, c, s in action_list:
                action_name = f'{final_action.__name__}_{c}_{s}'
                action_cnt = action_cnt_dict.get(action_name, 0)
                if action_cnt < max_repeat:
                    break

            action_cnt_dict[action_name] = action_cnt + 1
            invalid_queue += final_action(c, s, modify=True)

            # print(f'Action: {final_action.__name__}, Column: {c}, Stage: {s}, Count: {action_cnt}')
            # print(f'Invalid queue: {invalid_queue}')
            # for action in action_list:
            #     print(f'Candidate action: {action[2].__name__}, Column: {action[3]}, Stage: {action[4]}, Invalid assignment: {action[0]}')

            n_trial += 1
            if n_trial > max_trial:
                raise LegalizationError(f'Cannot legalize stage array after max_trial={max_trial}')

        return self.stage_array, n_trial
