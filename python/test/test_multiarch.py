import ducc0


def _level_number(level):
    if level == "x86-64":
        return 1
    prefix = "x86-64-v"
    assert level.startswith(prefix), level
    return int(level[len(prefix):])


def test_cpu_info_metadata_and_selector():
    info = ducc0.misc.cpu_info()
    if not info["multiarch"]:
        assert set(info) == {"architecture", "multiarch"}
        assert info["multiarch"] is False
        return

    assert set(info) == {
        "architecture",
        "multiarch",
        "compiled_levels",
        "available_levels",
        "max_level",
        "selected_level",
    }
    assert info["architecture"] == "x86-64"
    compiled = info["compiled_levels"]
    assert compiled == ["x86-64", "x86-64-v3", "x86-64-v4"]

    available = info["available_levels"]
    assert available
    assert available == compiled[:len(available)]
    assert available[0] == "x86-64"

    max_number = _level_number(info["max_level"])
    assert 1 <= max_number <= 4
    expected = next(level for level in reversed(available)
                    if _level_number(level) <= max_number)
    assert info["selected_level"] == expected
