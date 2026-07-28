"""Quick-look script for inspecting a ROOT file with uproot.

Usage:
    python scripts/Read_Root_File.py path/to/file.root
    python scripts/Read_Root_File.py path/to/file.root --tree AnalysisMiniTree
    python scripts/Read_Root_File.py path/to/file.root --tree AnalysisMiniTree --branches met_met tau1_pt --n 20
"""

import argparse

import uproot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to the .root file")
    parser.add_argument("--tree", default=None, help="Tree name (default: first TTree found)")
    parser.add_argument("--branches", nargs="+", default=None, help="Branches to load (default: all)")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to preview (default: 10)")
    args = parser.parse_args()

    with uproot.open(args.file) as f:
        classnames = f.classnames()
        tree_keys = [k for k, cls in classnames.items() if cls.startswith("TTree")]

        if not tree_keys:
            print("No TTrees in this file. Contents:")
            for key, cls in classnames.items():
                print(f"  {key}: {cls}")
            return

        print(f"TTrees found: {tree_keys}")
        tree_name = args.tree or tree_keys[0].split(";")[0]
        tree = f[tree_name]

        print(f"\nUsing tree: {tree_name!r}  ({tree.num_entries} entries)")
        print(f"Branches: {tree.keys()}")

        df = tree.arrays(args.branches, library="pd")
        print(f"\nPreview (first {args.n} rows):")
        print(df.head(args.n))


if __name__ == "__main__":
    main()
