# This code is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import subprocess
import sys
import sysconfig

import pytest


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="Requires free-threaded Python",
)
def test_import_preserves_free_threading():
    code = (
        "import sys\n"
        "assert not sys._is_gil_enabled(), 'GIL enabled before import'\n"
        "import ducc0\n"
        "assert not sys._is_gil_enabled(), 'DUCC enabled the GIL'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
