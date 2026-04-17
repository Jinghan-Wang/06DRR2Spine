#!/usr/bin/env python3
"""
List all wandb-related modules for PyInstaller hidden imports.
Run: python3 list_wandb_modules.py
"""

import pkgutil
import sys


def list_modules(package_name):
    print(f"\n{'='*60}")
    print(f"  Modules in: {package_name}")
    print(f"{'='*60}")
    try:
        pkg = __import__(package_name)
        print(f"  Location : {pkg.__file__}")
        modules = []
        for m in pkgutil.walk_packages(pkg.__path__, prefix=package_name + "."):
            modules.append(m.name)
            print(f"  {m.name}")
        print(f"\n  Total: {len(modules)} modules found")
        return modules
    except ImportError as e:
        print(f"  [SKIP] Cannot import {package_name}: {e}")
        return []
    except Exception as e:
        print(f"  [ERROR] {package_name}: {e}")
        return []


def main():
    targets = [
        "wandb",
        "wandb_gql",
        "wandb_core",
    ]

    all_modules = []
    for pkg in targets:
        mods = list_modules(pkg)
        all_modules.extend(mods)

    # Output as -i arguments for build.sh
    print(f"\n{'='*60}")
    print("  Copy-paste into build.sh command:")
    print(f"{'='*60}")
    for m in sorted(set(all_modules)):
        print(f'  -i "{m}" \\')

    # Also save to file
    out_file = "wandb_hidden_imports.txt"
    with open(out_file, "w") as f:
        for m in sorted(set(all_modules)):
            f.write(f'-i "{m}" \\\n')
    print(f"\n  Results also saved to: {out_file}")


if __name__ == "__main__":
    main()