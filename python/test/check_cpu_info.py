#!/usr/bin/env python3
"""Print and validate the CPU metadata exposed by an installed DUCC build."""

import argparse
import json
import os
import sys


PROFILE_LEVELS = {
    "x86-64": 1,
    "x86-64-v2": 2,
    "x86-64-v3": 3,
    "x86-64-v4": 4,
}
BASE_KEYS = {"architecture", "features", "multiarch"}
MULTIARCH_KEYS = BASE_KEYS | {
    "compiled_profiles",
    "available_profiles",
    "configured_limit",
    "active_profile",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_cpu_info(info, args):
    require(isinstance(info, dict), "cpu_info() must return a dictionary")
    require(set(info) == BASE_KEYS or set(info) == MULTIARCH_KEYS,
            "cpu_info() returned unexpected metadata fields")

    architecture = info.get("architecture")
    require(isinstance(architecture, str) and architecture,
            "architecture must be a nonempty string")
    features = info.get("features")
    require(isinstance(features, list), "features must be a list")
    require(all(isinstance(feature, str) and feature for feature in features),
            "features must contain nonempty strings")
    require(len(features) == len(set(features)),
            "features must not contain duplicates")
    multiarch = info.get("multiarch")
    require(isinstance(multiarch, bool), "multiarch must be a boolean")

    if architecture == "x86-64":
        if sys.platform.startswith("linux"):
            require("sse2" in features,
                    "Linux x86-64 must report its SSE2 baseline")
        require("avx2" not in features or "avx" in features,
                "avx2 was reported without avx")
        require("avx512" not in features or "avx" in features,
                "avx512 was reported without avx")
        require("avx512" not in features or "avx2" in features,
                "avx512 was reported without avx2")
    elif architecture == "aarch64":
        require(set(features) <= {"neon", "sve", "sve2"},
                "aarch64 reported an unknown CPU feature")

    require(not args.require_multiarch or multiarch,
            "multiarch=True is required for this check")
    if not multiarch:
        require(set(info) == BASE_KEYS,
                "non-multiarch cpu_info() returned an invalid metadata schema")
        require(args.require_active_profile is None,
                "an active profile can only be required for a multiarch build")
        require(not args.write_github_output,
                "profile eligibility outputs require a multiarch build")
        return

    require(set(info) == MULTIARCH_KEYS,
            "multiarch cpu_info() returned an invalid metadata schema")
    require(architecture == "x86-64",
            "multiarch metadata is only valid for x86-64")

    compiled = info["compiled_profiles"]
    expected_compiled = ["x86-64", "x86-64-v3", "x86-64-v4"]
    require(compiled == expected_compiled,
            "compiled_profiles must contain x86-64, x86-64-v3, and x86-64-v4")

    available = info["available_profiles"]
    require(isinstance(available, list),
            "available_profiles must be a list")
    require(all(isinstance(profile, str) and profile in PROFILE_LEVELS
                for profile in available),
            "available_profiles contains an unknown profile")
    require(available == expected_compiled[:len(available)] and available,
            "available_profiles must be an ordered prefix including x86-64")

    configured_limit = info["configured_limit"]
    require(isinstance(configured_limit, str)
            and configured_limit in PROFILE_LEVELS,
            "configured_limit must name an x86-64 profile")
    expected_active = max(
        (profile for profile in available
         if PROFILE_LEVELS[profile] <= PROFILE_LEVELS[configured_limit]),
        key=PROFILE_LEVELS.__getitem__,
    )
    active_profile = info["active_profile"]
    require(active_profile == expected_active,
            "active_profile is inconsistent with available_profiles and configured_limit")
    require(args.require_active_profile is None
            or active_profile == args.require_active_profile,
            "active_profile is {!r}; expected {!r}".format(
                active_profile, args.require_active_profile))

    if args.write_github_output:
        output_path = os.environ.get("GITHUB_OUTPUT")
        require(output_path, "GITHUB_OUTPUT is not set")
        with open(output_path, "a", encoding="utf-8") as output:
            output.write("v3={}\n".format(
                "true" if "x86-64-v3" in available else "false"))
            output.write("v4={}\n".format(
                "true" if "x86-64-v4" in available else "false"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-multiarch", action="store_true")
    parser.add_argument("--require-active-profile", choices=PROFILE_LEVELS)
    parser.add_argument("--write-github-output", action="store_true")
    args = parser.parse_args()

    try:
        import ducc0

        misc = getattr(ducc0, "misc", None)
        cpu_info = getattr(misc, "cpu_info", None)
        require(callable(cpu_info), "ducc0.misc.cpu_info() is unavailable")
        info = cpu_info()
        print(json.dumps(info, indent=2, sort_keys=True))
        check_cpu_info(info, args)
    except Exception as error:
        print("check_cpu_info.py: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
