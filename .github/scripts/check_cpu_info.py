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
        "compiled_profiles",
        "available_profiles",
        "configured_limit",
        "active_profile",
    }, info
    assert info["multiarch"] is True, info
    assert info["architecture"] == "x86-64", info
    assert info["compiled_profiles"] == [
        "x86-64",
        "x86-64-v3",
        "x86-64-v4",
    ], info
    available = info["available_profiles"]
    assert "x86-64" in available, info
    assert all(profile in info["compiled_profiles"] for profile in available), info
    assert available == info["compiled_profiles"][:len(available)], info
    assert info["configured_limit"] == "x86-64-v4", info
    assert info["active_profile"] == available[-1], info
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
            print(f"v3={'true' if 'x86-64-v3' in available else 'false'}", file=out)
            print(f"v4={'true' if 'x86-64-v4' in available else 'false'}", file=out)
else:
    raise ValueError(f"unknown build mode: {mode}")
