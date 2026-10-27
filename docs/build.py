"""Build the documentation from existing guides, source comments and CLI help."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ["README.md", "python/README.md", "octave/README.md", "CONTRIBUTING.md",
          "benchmarks/README.md", "docs/api.md", "docs/precision.md", "docs/extending.md",
          "docs/usage.md", "docs/building.md", "docs/serialization.md", "docs/performance.md"]
PAGES = {name: name for name in GUIDES}
PAGES.update({"README.md": "_readme.md", "docs/index.md": "index.md",
              "docs/reference/cpp.md": "docs/reference/cpp.md",
              "docs/reference/python.md": "docs/reference/python.md"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", nargs="?", type=Path, default=ROOT / "build/apxchol")
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "build/docs")
    parser.add_argument("--benchmarks", type=Path, help="directory of hosted benchmark JSON reports")
    args = parser.parse_args()
    cli, output = args.cli.resolve(), args.output.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="apxchol-docs-") as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        from performance import render
        reports = sorted(args.benchmarks.glob("*.json")) if args.benchmarks else []
        render(reports, source / "docs/performance", revision)
        for asset in (ROOT / "docs/performance").glob("*"):
            shutil.copyfile(asset, source / "docs/performance" / asset.name)
        for original, destination in PAGES.items():
            def link(match):
                href = match.group(1)
                parsed = urlsplit(href)
                # The Python package README also serves as the PyPI description.
                # Its absolute site links should stay local in HTML previews.
                site = "https://algoptgroup.github.io/apxchol/"
                if href.startswith(site):
                    path = parsed.path.removeprefix("/apxchol/")
                    page = path.removesuffix(".html") + ".md" if path else "index.md"
                    if page not in PAGES.values():
                        raise ValueError(f"Unknown documentation page: {href}")
                    url = os.path.relpath(source / page, (source / destination).parent)
                    return "](" + url + ("#" + parsed.fragment if parsed.fragment else "") + ")"
                if parsed.scheme or parsed.netloc or not parsed.path:
                    return match.group(0)
                target = (ROOT / original).parent / parsed.path
                relative = target.resolve().relative_to(ROOT).as_posix()
                if relative in PAGES:
                    page = "index.md" if relative == "README.md" else PAGES[relative]
                    url = os.path.relpath(source / page, (source / destination).parent)
                elif relative.startswith("docs/performance/"):
                    url = os.path.relpath(source / relative, (source / destination).parent)
                else:
                    if not target.exists():
                        raise FileNotFoundError(target)
                    kind = "tree" if target.is_dir() else "blob"
                    url = f"https://github.com/AlgOptGroup/apxchol/{kind}/{revision}/{relative}"
                if parsed.fragment:
                    url += "#" + parsed.fragment
                return "](" + url + ")"

            text = re.sub(r"\]\(([^\s)]+)\)", link, (ROOT / original).read_text())
            path = source / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        help_text = subprocess.check_output(["apxchol", "--help"], executable=str(cli), cwd=ROOT, text=True)
        (source / "docs/reference/cli.md").write_text(
            "# Command-line reference\n\n"
            "See [build and run](../../index.md#build-and-run) for examples. "
            "Options below come from this revision's executable.\n\n```text\n" + help_text + "```\n")
        env = dict(os.environ, APXCHOL_DOCS_BUILD=str(work))
        subprocess.run(["doxygen", str(ROOT / "docs/Doxyfile")], cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html",
                        "-d", str(work / "doctrees"), "-c", str(ROOT / "docs"), str(source), str(output)], env=env, check=True)


if __name__ == "__main__":
    main()
