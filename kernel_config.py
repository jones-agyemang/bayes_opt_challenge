from dataclasses import dataclass

@dataclass
class KernelConfig:
    """Class for keeping kernel configurations"""
    MATERN = 'Matern'
    RBF = 'RBF'
    VALID_KERNEL_TYPES = { RBF, MATERN }
    DEFAULT_LEN_SCALE = 0.3
    DEFAULT_LEN_SCALE_BOUND = (1e-2, 1e2)

    kernel_type:  str
    length_scale: float = DEFAULT_LEN_SCALE
    length_scale_bounds: (float, float) = DEFAULT_LEN_SCALE_BOUND
    mu: float = None

    def __post_init__(self):
        self.set_kernel_type(self.kernel_type)
        self.set_mu(self.mu)

    def set_kernel_type(self, kernel_type: str):
        if kernel_type not in self.VALID_KERNEL_TYPES:
            raise(TypeError(f'Invalid kernel type: {kernel_type}'))
        self._kernel_type = kernel_type

    def set_mu(self, mu: float):
        match self.kernel_type:
            case 'Matern':
                if mu is None:
                    raise(ValueError("Value of 'mu' cannot be null"))
        #if kernel_type == 'Matern':
        #    if mu is None:
        #        raise(ValueError(f'Incorrect value: {mu}'))
