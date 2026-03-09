import pytest
from kernel_config import ( KernelConfig )

def test_sets_valid_kernel_type():
    kernel_cfg = KernelConfig(kernel_type='RBF')
    assert kernel_cfg.kernel_type, 'RBF'

def test_rejects_invalid_kernel_type():
    with pytest.raises(TypeError):
        KernelConfig(kernel_type='invalid_type')

def test_sets_default_values():
    kernel_cfg = KernelConfig(kernel_type='Matern', mu=3.4)
    assert kernel_cfg.length_scale, DEFAULT_LEN_SCALE
    assert kernel_cfg.length_scale_bounds, DEFAULT_LEN_SCALE_BOUND

def test_presence_of_kernel_smoothness_value():
    with pytest.raises(ValueError):
        kernel_cfg = KernelConfig(kernel_type='Matern')