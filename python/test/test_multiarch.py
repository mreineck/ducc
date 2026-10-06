import ducc0


def _profile_number(profile):
    if profile == "x86-64":
        return 1
    prefix = "x86-64-v"
    assert profile.startswith(prefix), profile
    return int(profile[len(prefix):])


def test_cpu_info_metadata_and_selector():
    info = ducc0.misc.cpu_info()
    if not info["multiarch"]:
        assert set(info) == {"architecture", "multiarch"}
        assert info["multiarch"] is False
        return

    assert set(info) == {
        "architecture",
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
