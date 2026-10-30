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
GUIDES = ["README.md", "CHANGELOG.md", "python/README.md", "octave/README.md", "CONTRIBUTING.md",
          "benchmarks/README.md", "docs/api.md", "docs/precision.md", "docs/extending.md",
          "docs/usage.md", "docs/building.md", "docs/batching.md", "docs/serialization.md",
          "docs/sdk.md", "docs/performance.md", "docs/install.md", "docs/quickstart.md",
          "docs/cli.md", "docs/cpp.md", "docs/examples.md", "docs/citing.md"]
PAGES = {name: name for name in GUIDES}
PAGES.update({"README.md": "_readme.md", "docs/index.md": "index.md",
              "docs/reference/cpp.md": "docs/reference/cpp.md",
              "docs/reference/python.md": "docs/reference/python.md"})
CLI_REFERENCE = "docs/reference/cli.md"  # generated from --help below


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", nargs="?", type=Path, default=ROOT / "build/apxchol")
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "build/docs")
    args = parser.parse_args()
    cli, output = args.cli.resolve(), args.output.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="apxchol-docs-") as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        (source / "docs/performance").mkdir(parents=True)
        for asset in (ROOT / "docs/performance").glob("*"):
            shutil.copyfile(asset, source / "docs/performance" / asset.name)
        shutil.copytree(ROOT / "examples", source / "examples")
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
                if relative in PAGES or relative == CLI_REFERENCE:
                    page = "index.md" if relative == "README.md" else PAGES.get(relative, relative)
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
        (source / CLI_REFERENCE).write_text(
            "# Command-line reference\n\n"
            "See the [command-line guide](../cli.md) for examples. "
            "Options below come from this revision's executable.\n\n```text\n" + help_text + "```\n")
        env = dict(os.environ, APXCHOL_DOCS_BUILD=str(work))
        subprocess.run(["doxygen", str(ROOT / "docs/Doxyfile")], cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html",
                        "-d", str(work / "doctrees"), "-c", str(ROOT / "docs"), str(source), str(output)], env=env, check=True)


if __name__ == "__main__":
    main()
