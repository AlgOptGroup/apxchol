"""Assemble Pages from development HTML and immutable release documentation assets."""
import argparse
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

SELECTOR_START = "<!-- apxchol-version-selector -->"
SELECTOR_END = "<!-- /apxchol-version-selector -->"
TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def check_tag(tag):
    if not isinstance(tag, str) or not TAG.fullmatch(tag):
        raise ValueError(f"Invalid documentation release tag: {tag!r}")


def snapshot_releases(pages):
    """Return stable tags with documentation assets, newest publication first."""
    releases = []
    seen = set()
    for page in pages:
        for release in page:
            if release["draft"] or release["prerelease"]:
                continue
            assets = release["assets"]
            if not isinstance(assets, list) or any(not isinstance(a, dict) for a in assets):
                raise ValueError("Malformed release assets")
            matches = [a for a in assets if a.get("name") == "documentation.tar.gz"]
            if not matches:
                continue
            tag = release["tag_name"]
            check_tag(tag)
            if len(matches) != 1 or tag in seen:
                raise ValueError(f"Duplicate documentation snapshot: {tag}")
            published = datetime.fromisoformat(release["published_at"].replace("Z", "+00:00"))
            if published.tzinfo is None:
                raise ValueError(f"Missing publication timezone: {tag}")
            seen.add(tag)
            releases.append((published, tag))
    return [tag for _, tag in sorted(releases, reverse=True)]


def extract_snapshot(archive, destination):
    """Extract only ordinary documentation files within a fresh directory."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or not (member.isfile() or member.isdir()) or path in seen):
                raise ValueError(f"Unsafe or duplicate documentation path: {member.name!r}")
            seen.add(path)
        if not any(PurePosixPath(m.name) == PurePosixPath("index.html") and m.isfile()
                   for m in members):
            raise ValueError("Documentation archive has no root index.html")
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def assemble(development, snapshots, output):
    """Copy snapshots and fill their marked selectors; keep source trees unchanged."""
    versions = [(tag, source) for tag, source in snapshots] + [("Development", development)]
    tags = [tag for tag, _ in snapshots]
    for tag in tags:
        check_tag(tag)
    if len(set(tags)) != len(tags):
        raise ValueError("Duplicate documentation release tags")
    for _, source in versions:
        if not (source / "index.html").is_file():
            raise ValueError(f"Documentation has no root index.html: {source}")
    current, stable = versions[0] if snapshots else versions[-1]
    shutil.copytree(stable, output)
    destinations = [(current, output)]
    for label, source in versions:
        destination = output / ("dev" if label == "Development" else label)
        shutil.copytree(source, destination)
        destinations.append((label, destination))
    targets = [(label, destination / "index.html") for label, destination in destinations[1:]]
    # Enumerate each tree separately so the root alias does not rewrite nested versions.
    for label, destination in destinations:
        for page in destination.rglob("*.html"):
            if destination == output and any(parent in page.parents for _, parent in destinations[1:]):
                continue
            html = page.read_bytes().decode("utf-8")
            if SELECTOR_START not in html and SELECTOR_END not in html:
                continue  # Older snapshots remain linked from the current site's selector.
            if html.count(SELECTOR_START) != 1 or html.count(SELECTOR_END) != 1:
                raise ValueError(f"Malformed version selector: {page}")
            before, remainder = html.split(SELECTOR_START)
            _, after = remainder.split(SELECTOR_END)
            links = []
            for name, target in targets:
                href = escape(os.path.relpath(target, page.parent), quote=True)
                selected = ' aria-current="page"' if name == label else ""
                links.append(f'<li><a href="{href}"{selected}>{name}</a></li>')
            selector = (f'\n<details class="version-selector">\n'
                        f'  <summary aria-label="Documentation version">{label}</summary>\n'
                        f'  <ul>{"".join(links)}</ul>\n</details>\n')
            page.write_bytes((before + SELECTOR_START + selector + SELECTOR_END + after).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("releases", type=Path, help="gh api --paginate --slurp release listing")
    parser.add_argument("repository", help="GitHub owner/repository")
    args = parser.parse_args()
    tags = snapshot_releases(json.loads(args.releases.read_text()))
    with tempfile.TemporaryDirectory(prefix="apxchol-docs-releases-") as temporary:
        snapshots = []
        for tag in tags:
            download = Path(temporary) / tag
            download.mkdir()
            subprocess.run(["gh", "release", "download", tag, "--pattern", "documentation.tar.gz",
                            "--dir", str(download), "--repo", args.repository], check=True)
            extracted = download / "html"
            extract_snapshot(download / "documentation.tar.gz", extracted)
            snapshots.append((tag, extracted))
        assemble(args.development, snapshots, args.output)


if __name__ == "__main__":
    main()
