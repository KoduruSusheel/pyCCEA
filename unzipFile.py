import argparse
import os
import zipfile
import shutil
from typing import Optional, Tuple, Set

# Canonical dataset folder names expected under ./data/
CANONICAL_NAMES = {
    'uc_merced': 'UCMerced_LandUse',
    'lc25000': 'lung_colon_image_set',
    'plants': 'split_ttv_dataset_type_of_plants',
}

# Heuristic token mapping from filename to dataset key
TOKEN_MAP = [
    (('ucmerced', 'uc', 'merced', 'landuse'), 'uc_merced'),
    (('lc25000', 'lung', 'colon', 'lung_colon'), 'lc25000'),
    (('plants', 'split_ttv', 'type', 'ttv'), 'plants'),
]

def infer_dataset_key(name: str) -> Optional[str]:
    lower = name.lower()
    for tokens, key in TOKEN_MAP:
        if any(tok in lower for tok in tokens):
            return key
    return None


def top_level_dirs(z: zipfile.ZipFile) -> Tuple[Set[str], bool]:
    roots = set()
    for n in z.namelist():
        parts = n.split('/') if '/' in n else n.split('\\')
        if parts and parts[0]:
            roots.add(parts[0])
    # Consider it a single-rooted archive if exactly one top-level dir and it's a directory-like entry
    return roots, len(roots) == 1


def extract_zip(zip_path: str, data_dir: str, force: bool) -> None:
    base_name = os.path.basename(zip_path)
    name_no_ext = os.path.splitext(base_name)[0]
    dataset_key = infer_dataset_key(name_no_ext) or 'unknown'
    canonical = CANONICAL_NAMES.get(dataset_key)

    os.makedirs(data_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        roots, single_root = top_level_dirs(z)
        # Decide target behavior
        if single_root:
            # Extract into data/, then optionally rename the single root to canonical
            print(f"Extracting '{base_name}' into '{data_dir}' (single-root archive: {list(roots)[0]}) ...")
            z.extractall(data_dir)
            root_name = list(roots)[0]
            src_path = os.path.join(data_dir, root_name)
            if canonical and root_name != canonical:
                dst_path = os.path.join(data_dir, canonical)
                if os.path.exists(dst_path):
                    if force:
                        print(f"--force: removing existing '{dst_path}' before rename")
                        shutil.rmtree(dst_path)
                    else:
                        print(f"Skipping rename: target '{dst_path}' exists. Keeping original '{src_path}'.")
                        return
                try:
                    os.replace(src_path, dst_path)
                    print(f"Renamed '{root_name}' -> '{canonical}'")
                except Exception as e:
                    print(f"Warning: could not rename '{src_path}' to '{dst_path}': {e}")
        else:
            # Multi-root: extract into a dedicated folder to avoid clutter
            target_folder = canonical or name_no_ext
            dest = os.path.join(data_dir, target_folder)
            if os.path.exists(dest):
                if force:
                    print(f"--force: removing existing '{dest}' before extract")
                    shutil.rmtree(dest)
                else:
                    print(f"Skipping '{base_name}': target folder '{dest}' already exists. Use --force to re-extract.")
                    return
            os.makedirs(dest, exist_ok=True)
            print(f"Extracting '{base_name}' into '{dest}' (multi-root archive) ...")
            z.extractall(dest)


def main():
    parser = argparse.ArgumentParser(description="Extract known dataset ZIPs into ./data/")
    parser.add_argument('--source', type=str, default='.', help='Directory to scan for .zip files (default: project root)')
    parser.add_argument('--force', action='store_true', help='Re-extract even if target folder exists (or rename conflicts)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.abspath(args.source if os.path.isabs(args.source) else os.path.join(project_root, args.source))
    data_dir = os.path.join(project_root, 'data')

    print(f"Scanning for .zip files in: {source_dir}")
    zips = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith('.zip')]

    if not zips:
        print("No .zip files found. Place dataset zips in project root or pass --source.")
        return

    os.makedirs(data_dir, exist_ok=True)

    for zp in zips:
        try:
            extract_zip(zp, data_dir, args.force)
        except zipfile.BadZipFile:
            print(f"Skipping '{zp}': not a valid zip file.")
        except Exception as e:
            print(f"Error extracting '{zp}': {e}")

    print("\nDone. Expected structures after success:")
    print("- data/UCMerced_LandUse/Images")
    print("- data/lung_colon_image_set/Train and Validation Set, Test Set")
    print("- data/split_ttv_dataset_type_of_plants/Train_Set_Folder, Validation_Set_Folder, Test_Set_Folder")

if __name__ == '__main__':
    main()
