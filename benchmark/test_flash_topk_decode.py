# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest

from tests.test_flash_topk import pytestmark  # noqa: F401

from .flash_topk_benchmark import flash_topk_benchmark


@pytest.mark.flash_topk_decode
def test_flash_topk_decode():
    flash_topk_benchmark(prefill=False).run()
