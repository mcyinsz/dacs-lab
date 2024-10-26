import abc
import random
from collections import OrderedDict

class BaseDesignParam(abc.ABC):
    """
        Base class for parameters in design space.
    """

    def __init__(self,
                 name: str,
                 description: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.description = description

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description
    
    @abc.abstractmethod
    def get_size(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_val_by_idx(self, idx):
        raise NotImplementedError


class CategoricalDesignParam(BaseDesignParam):

    def __init__(self,
                 name: str,
                 choices: list,
                 description: str | None = None) -> None:
        super().__init__(name, description)
        self.choices = choices

    def get_size(self):
        return len(self.choices)

    def get_val_by_idx(self, idx: int):
        if idx < 0 or idx >= len(self.choices):
            raise ValueError("Index out of range.")
        return self.choices[idx]


class BaseDesignSpace():
    """
        Base class for design space.
    """

    def __init__(self, configs: list) -> None:
        super().__init__()
        self._space = OrderedDict()
        self.init_design_space(configs)

    def init_design_space(self, configs: list) -> None:
        for param_config in configs:
            param_name = param_config["name"]
            param_type = param_config["type"]
            param_desc = param_config.get('description', None)
            
            if param_type == 'categorical':
                param_choices = param_config["choices"]
                param = CategoricalDesignParam(param_name, param_choices, param_desc)
            else:
                raise NotImplementedError(f"Unsupported parameter type: {param_type}")
            
            self._space[param_name] = param

    def get_dims(self) -> int:
        """
            Number of dimensions in the design space.
        """
        return len(self._space)
    
    def get_size(self) -> dict:
        """
            Sizes of each parameter.
        """
        return {param_name: param.get_size() for param_name, param in self._space.items()}
    
    def generate_design_point_by_random(self, seed=1234) -> OrderedDict:
        """
            Generate design point randomly until the first valid one.
            Returns: design point
        """
        random.seed(seed)

        def sample_design_point():
            design_point = OrderedDict()

            for param_name, param in self._space.items():
                param_idx = random.randint(0, param.get_size() - 1)
                param_val = param.get_val_by_idx(param_idx)
                design_point[param_name] = param_val

            return design_point

        while True:
            design_point = sample_design_point()
            if self.validate_design_point(design_point):
                break

        return design_point
    
    def validate_design_point(self, design_point: OrderedDict) -> bool:
        """
            Validate a design point.
            Returns: True if valid, False otherwise.
        """
        # We skip the validation process for parameter tuning
        return True
    

CADENCE_GENUS_DESIGN_SPACE = [
    {
        'name': 'clk_period_ns',
        'type': 'categorical',
        'choices': [2.5, 5.0, 7.5, 10.0],
    },
    {
        'name': 'syn_generic_effort',
        'type': 'categorical',
        'choices': ['low', 'medium', 'high'],
    },
    {
        'name': 'syn_map_effort',
        'type': 'categorical',
        'choices': ['low', 'medium', 'high'],
    },
    {
        'name': 'syn_opt_effort',
        'type': 'categorical',
        'choices': [None, 'low', 'medium', 'high'],
    },
    {
        'name': 'max_fanout',
        'type': 'categorical',
        'choices': [None, 10, 20, 30],
    },
    {
        'name': 'max_transition_ns',
        'type': 'categorical',
        'choices': [None, 0.1, 0.2, 0.3],
    },
    {
        'name': 'max_capacitance_ff',
        'type': 'categorical',
        'choices': [None, 0.1, 0.2, 0.3],
    }
]


CADENCE_INNOVUS_DESIGN_SPACE = [
    {
        'name': 'place_utilization',
        'type': 'categorical',
        'choices': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    },
    {
        'name': 'place_detail_eco_priority_insts',
        'type': 'categorical',
        'choices': ['placed', 'fixed', 'eco'],
    },
    {
        'name': 'place_detail_activity_power_driven',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
    {
        'name': 'place_detail_wire_length_opt_effort',
        'type': 'categorical',
        'choices': ['low', 'medium', 'high'],
    },
    {
        'name': 'place_global_auto_blockage_in_channel',
        'type': 'categorical',
        'choices': ['none', 'soft', 'partial'],
    },
    {
        'name': 'place_global_activity_power_driven',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
    {
        'name': 'place_global_activity_power_driven_effort',
        'type': 'categorical',
        'choices': ['standard', 'high'],
    },
    {
        'name': 'place_global_clock_power_driven',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
    {
        'name': 'place_global_clock_power_driven_effort',
        'type': 'categorical',
        'choices': ['low', 'standard', 'high'],
    },
    {
        'name': 'place_global_timing_effort',
        'type': 'categorical',
        'choices': ['medium', 'high'],
    },
    {
        'name': 'place_global_cong_effort',
        'type': 'categorical',
        'choices': ['low', 'medium', 'high', 'extreme', 'auto'],
    },
    {
        'name': 'place_global_clock_gate_aware',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
    {
        'name': 'place_global_uniform_density',
        'type': 'categorical',
        'choices': ['true', 'false'],
    },
]