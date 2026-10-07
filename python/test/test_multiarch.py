import os
import subprocess
import sys

import ducc0


def _profile_number(profile):
    if profile == "x86-64":
        return 1
    prefix = "x86-64-v"
    assert profile.startswith(prefix), profile
    return int(profile[len(prefix):])


def test_cpu_info_metadata_and_selector():
    assert "features" in (ducc0.misc.cpu_info.__doc__ or "")
    info = ducc0.misc.cpu_info()
    assert isinstance(info.get("architecture"), str)
    assert info["architecture"]
    assert isinstance(info.get("features"), list)
    assert all(isinstance(feature, str) for feature in info["features"])
    assert isinstance(info.get("multiarch"), bool)

    features = info["features"]
    if info["architecture"] == "x86-64":
        if sys.platform.startswith("linux"):
            assert "sse2" in features
        assert "avx2" not in features or "avx" in features
        assert "avx512" not in features or "avx" in features
    elif info["architecture"] == "aarch64":
        assert set(features) <= {"neon", "sve", "sve2"}

    if not info["multiarch"]:
        assert set(info) == {"architecture", "features", "multiarch"}
        assert info["multiarch"] is False
        return

    assert set(info) == {
        "architecture",
        "features",
        "multiarch",
        "compiled_profiles",
        "available_profiles",
        "configured_limit",
        "active_profile",
    }
    assert info["architecture"] == "x86-64"
    compiled = info["compiled_profiles"]
    assert compiled == ["x86-64", "x86-64-v3", "x86-64-v4"]

    available = info["available_profiles"]
    assert available
    assert available == compiled[:len(available)]
    assert available[0] == "x86-64"

    limit_number = _profile_number(info["configured_limit"])
    assert 1 <= limit_number <= 4
    expected = next(profile for profile in reversed(available)
                    if _profile_number(profile) <= limit_number)
    assert info["active_profile"] == expected


def test_malformed_configured_limit_fails_clearly():
    if not ducc0.misc.cpu_info()["multiarch"]:
        return

    env = os.environ.copy()
    env["DUCC0_MAX_PSABI_LEVEL"] = "3invalid"
    result = subprocess.run(
        [sys.executable, "-c", "import ducc0"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "DUCC0_MAX_PSABI_LEVEL" in result.stderr
