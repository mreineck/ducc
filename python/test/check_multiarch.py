#!/usr/bin/env python3
"""Print and validate the CPU metadata exposed by an installed DUCC build."""

# This code is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this code; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# Copyright (C) 2026 Albert M. W. Yau
# Author: Albert M. W. Yau

import argparse
import os
import sys


PROFILE_LEVELS = {
    "x86-64": 1,
    "x86-64-v2": 2,
    "x86-64-v3": 3,
    "x86-64-v4": 4,
}
PROFILE_LEVELS_2 = {
    1: "x86-64",
    2: "x86-64-v2",
    3: "x86-64-v3",
    4: "x86-64-v4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_multiarch_support(info, args):
    require(isinstance(info, int), "psabi_level() must return an integer")
    multiarch = info>0

    require(not args.require_multiarch or multiarch,
            "multiarch=True is required for this check")
    if not multiarch:
        require(args.require_active_profile is None,
                "an active profile can only be required for a multiarch build")
        require(not args.write_github_output,
                "profile eligibility outputs require a multiarch build")
        return

    require(args.require_active_profile is None
            or PROFILE_LEVELS_2[info] == args.require_active_profile,
            "active_profile is {!r}; expected {!r}".format(
                PROFILE_LEVELS_2[info], args.require_active_profile))

    if args.write_github_output:
        output_path = os.environ.get("GITHUB_OUTPUT")
        require(output_path, "GITHUB_OUTPUT is not set")
        with open(output_path, "a", encoding="utf-8") as output:
            output.write("v3={}\n".format(
                "true" if info >= 3 else "false"))
            output.write("v4={}\n".format(
                "true" if info >= 4 else "false"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-multiarch", action="store_true")
    parser.add_argument("--require-active-profile", choices=PROFILE_LEVELS)
    parser.add_argument("--write-github-output", action="store_true")
    args = parser.parse_args()

    try:
        import ducc0
        check_multiarch_support(ducc0.misc.psabi_level(), args)
    except Exception as error:
        print("check_multiarch.py: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
