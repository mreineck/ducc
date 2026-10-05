import json
import os
import sys

import ducc0


mode = sys.argv[1]
info = ducc0.misc.cpu_info()
print(json.dumps(info, indent=2))

if mode in ("native", "portable"):
    assert info["multiarch"] is False, info
    assert set(info) == {"architecture", "multiarch"}, info
elif mode == "multiarch":
    assert set(info) == {
        "architecture",
        "multiarch",
        "compiled_levels",
        "available_levels",
        "max_level",
        "selected_level",
    }, info
    assert info["multiarch"] is True, info
    assert info["architecture"] == "x86-64", info
    assert info["compiled_levels"] == [
        "x86-64",
        "x86-64-v3",
        "x86-64-v4",
    ], info
    available = info["available_levels"]
    assert "x86-64" in available, info
    assert all(level in info["compiled_levels"] for level in available), info
    assert info["selected_level"] in available, info
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
            print(f"v3={'true' if 'x86-64-v3' in available else 'false'}", file=out)
            print(f"v4={'true' if 'x86-64-v4' in available else 'false'}", file=out)
else:
    raise ValueError(f"unknown build mode: {mode}")
