from dataclasses import dataclass, field

@dataclass
class KernelConfig:
    """Class for keeping kernel configurations"""
    VALID_KERNEL_TYPES = { "RBF", "Matern" }

    kernel_type: str = "RBF"

    _kernel_type: str = field(init=False, repr=False)
    length_scale: float = 0.3
    length_scale_bounds: tuple[float, float] = (1e-2, 1e2)
    nu: float | None = None

    def __post_init__(self):
        self.kernel_type = self.kernel_type

        if self.kernel_type == "Matern" and self.nu is None:
            raise(ValueError("Value of 'nu' cannot be null for Matern"))

    @property
    def kernel_type(self) -> str:
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, value: str) -> None:
        if value not in self.VALID_KERNEL_TYPES:
            raise TypeError(f"Invalid kernel type: {value}")
        self._kernel_type = value

    def set_kernel_type(self, kernel_type: str):
        if kernel_type not in self.VALID_KERNEL_TYPES:
            raise(TypeError(f'Invalid kernel type: {kernel_type}'))
        self.kernel_type = kernel_type

    def set_nu(self, nu: float):
        if self.kernel_type == 'Matern':
            if nu is None:
                raise(ValueError("Value of 'nu' cannot be null"))
