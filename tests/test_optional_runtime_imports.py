# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import contracts run on every backend; CUDA communication is not simulated."""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.fused_allreduce_rms_norm


def _run(code):
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], check=True, timeout=120
    )


@pytest.mark.parametrize("vendor", ["thead", "hygon", "metax", "mthreads"])
def test_non_nvidia_import_never_initializes_mnnvl(vendor):
    _run(
        """
        import builtins
        import importlib.util
        import sys
        from types import SimpleNamespace
        import flaggems_vllm.runtime as runtime
        original_module = __import__('importlib').import_module(
            'flaggems_vllm.ops.fused_allreduce_rms_norm')
        runtime.device = SimpleNamespace(vendor_name=VENDOR)
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith('flashinfer.comm'):
                raise AssertionError('NVIDIA communication imported on ' + VENDOR)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded
        spec = importlib.util.spec_from_file_location('_mnnvl_non_nvidia', original_module.__file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert module._HAS_MNNVL_RUNTIME is False
        for name in ('TorchDistBackend', 'McastGPUBuffer', 'all_ranks_support_mnnvl',
                     'create_tensor_from_cuda_memory', 'is_multicast_supported'):
            assert getattr(module, name) is None
    """.replace(
            "VENDOR", repr(vendor)
        )
    )


@pytest.mark.parametrize(
    "outcome", ["available", "import_error", "os_error", "runtime_error"]
)
def test_nvidia_mnnvl_optional_dependency_contract(outcome):
    _run(
        """
        import builtins
        import importlib.util
        import sys
        from types import ModuleType, SimpleNamespace
        import flaggems_vllm.runtime as runtime
        original_module = __import__('importlib').import_module(
            'flaggems_vllm.ops.fused_allreduce_rms_norm')
        runtime.device = SimpleNamespace(vendor_name='nvidia')
        marker = object()
        outcome = OUTCOME
        attempts = []
        original_import = builtins.__import__
        for name in ('flashinfer', 'flashinfer.comm',
                     'flashinfer.comm.comm_backend', 'flashinfer.comm.mnnvl'):
            sys.modules[name] = ModuleType(name)
        sys.modules['flashinfer.comm.comm_backend'].TorchDistBackend = marker
        for name in ('McastGPUBuffer', 'all_ranks_support_mnnvl',
                     'create_tensor_from_cuda_memory', 'is_multicast_supported'):
            setattr(sys.modules['flashinfer.comm.mnnvl'], name, marker)
        def guarded(name, *args, **kwargs):
            if name == 'flashinfer.comm.mnnvl':
                attempts.append(name)
                errors = {'import_error': ImportError, 'os_error': OSError,
                          'runtime_error': RuntimeError}
                if outcome in errors:
                    raise errors[outcome]('optional-mnnvl-failure')
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded
        spec = importlib.util.spec_from_file_location('_mnnvl_nvidia', original_module.__file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        if outcome == 'runtime_error':
            try:
                spec.loader.exec_module(module)
            except RuntimeError as error:
                assert str(error) == 'optional-mnnvl-failure'
            else:
                raise AssertionError('unexpected import failure was swallowed')
        else:
            spec.loader.exec_module(module)
            available = outcome == 'available'
            assert module._HAS_MNNVL_RUNTIME is available
            for name in ('TorchDistBackend', 'McastGPUBuffer', 'all_ranks_support_mnnvl',
                         'create_tensor_from_cuda_memory', 'is_multicast_supported'):
                assert getattr(module, name) is (marker if available else None)
        assert attempts == ['flashinfer.comm.mnnvl']
    """.replace(
            "OUTCOME", repr(outcome)
        )
    )
