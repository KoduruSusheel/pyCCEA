import argparse
import subprocess
import sys
import os
from typing import Dict, Tuple

try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

# Minimal required versions known to work with this project
REQUIRED: Dict[str, str] = {
    'torch': '>=2.1',
    'torchvision': '>=0.16',
    'numpy': '>=1.22',
    'scikit-learn': '>=1.1',
    'pillow': '>=9.0',
    'matplotlib': '>=3.6',
    'seaborn': '>=0.12',
}

# Mapping for import name -> package name for metadata (when they differ)
METADATA_NAME = {
    'pillow': 'Pillow',
}


def is_installed(pkg: str) -> Tuple[bool, str]:
    meta_name = METADATA_NAME.get(pkg, pkg)
    try:
        v = version(meta_name)
        return True, v
    except PackageNotFoundError:
        return False, ''


def pip_install(args_list):
    cmd = [sys.executable, '-m', 'pip', 'install'] + args_list
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description='Check and install required Python packages for the project.')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be installed without making changes')
    parser.add_argument('--write-req', action='store_true', help='Write a requirements.txt in the project root')
    parser.add_argument('--cuda', type=str, default='', help='Optional CUDA tag for PyTorch (e.g., cu121). If omitted, CPU wheels are used.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    missing = []
    installed_info = {}

    print('Checking required packages...')
    for pkg, constraint in REQUIRED.items():
        ok, ver = is_installed(pkg)
        if ok:
            print(f"- {pkg} already installed (version {ver})")
            installed_info[pkg] = ver
        else:
            print(f"- {pkg} NOT installed (requires {constraint})")
            missing.append(pkg)

    if args.write_req:
        req_lines = []
        for pkg, constraint in REQUIRED.items():
            if pkg in installed_info and installed_info[pkg]:
                req_lines.append(f"{pkg}=={installed_info[pkg]}")
            else:
                req_lines.append(f"{pkg}{constraint}")
        req_path = os.path.join(project_root, 'requirements.txt')
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(req_lines) + '\n')
        print(f"Wrote requirements.txt -> {req_path}")

    if not missing:
        print('All required packages appear to be installed.')
        return

    if args.dry_run:
        print('\nDry run: would install ->', ', '.join(missing))
        return

    # Install torch and torchvision first with optional CUDA index if requested
    to_install = []
    if 'torch' in missing or 'torchvision' in missing:
        torch_pkgs = []
        if 'torch' in missing:
            torch_pkgs.append('torch')
        if 'torchvision' in missing:
            torch_pkgs.append('torchvision')
        if args.cuda:
            # Use PyTorch wheel index for specific CUDA tag
            rc = pip_install(torch_pkgs + ['--index-url', f'https://download.pytorch.org/whl/{args.cuda}'])
        else:
            rc = pip_install(torch_pkgs)
        if rc != 0:
            print('Warning: PyTorch installation command returned non-zero exit code. You may need to install manually based on your CUDA setup: https://pytorch.org/get-started/locally/')
        for p in torch_pkgs:
            if p in missing:
                missing.remove(p)

    if missing:
        rc = pip_install(missing)
        if rc != 0:
            print('One or more installations failed. Please review the output above.')

    print('Dependency check/installation complete.')


if __name__ == '__main__':
    main()
