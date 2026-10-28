#!/usr/bin/env python3
# usage: 
# python files_to_md.py --file-list files.txt -o codefiles.md
# python files_to_md.py a.py b.py --file-list files.txt

import argparse
from pathlib import Path
import sys

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".sh", ".go", ".rs",
    ".cpp", ".c", ".cs", ".rb", ".php"
}


def read_file_list(file_list_path):
    """Read file paths from a text file (one per line)."""
    paths = []
    path = Path(file_list_path)

    if not path.exists():
        print(f"[ERROR] File list not found: {file_list_path}", file=sys.stderr)
        return paths

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(line)

    return paths


def files_to_markdown(files, output_md):
    seen = set()

    with open(output_md, "w", encoding="utf-8") as out:
        out.write(f"# CURRENT FILES USED:\n\n")

        for file_path in files:
            path = Path(file_path)

            # De-duplicate while preserving order
            if path in seen:
                continue
            seen.add(path)

            if not path.exists():
                print(f"[WARN] Skipping missing file: {file_path}", file=sys.stderr)
                continue

            out.write(f"## {path}\n\n")

            content = path.read_text(encoding="utf-8")

            if path.suffix in CODE_EXTENSIONS:
                language = path.suffix.lstrip(".")
                out.write(f"```{language}\n")
                out.write(content.rstrip())
                out.write("\n```\n\n")
            else:
                out.write(content.rstrip())
                out.write("\n\n")

            out.write("---\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple files into a single Markdown file."
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Files to include directly"
    )

    parser.add_argument(
        "--file-list",
        help="Path to a .txt file containing a list of files (one per line)"
    )

    parser.add_argument(
        "-o", "--output",
        default="files_combined.md",
        help="Output Markdown file (default: files_combined.md)"
    )

    args = parser.parse_args()

    files = []

    if args.file_list:
        files.extend(read_file_list(args.file_list))

    if args.files:
        files.extend(args.files)

    if not files:
        print("[ERROR] No input files provided.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    files_to_markdown(files, args.output)
    print(f"[OK] Wrote Markdown file: {args.output}")


if __name__ == "__main__":
    main()
