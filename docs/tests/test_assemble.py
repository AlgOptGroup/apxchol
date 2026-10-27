"""Checks for assembling versioned documentation from existing snapshots."""
import importlib.util
import io
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import tarfile
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("assemble", Path(__file__).parents[1] / "assemble.py")
assembly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembly)


class Selector(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.summary = ""
        self.links = []
        self.in_summary = False
        self.in_link = False
        self.feed(html.split(assembly.SELECTOR_START, 1)[1].split(assembly.SELECTOR_END, 1)[0])

    def handle_starttag(self, tag, attrs):
        if tag == "summary":
            self.in_summary = True
        elif tag == "a":
            self.in_link = True
            self.links.append([dict(attrs), ""])

    def handle_endtag(self, tag):
        if tag == "summary":
            self.in_summary = False
        elif tag == "a":
            self.in_link = False

    def handle_data(self, text):
        if self.in_summary:
            self.summary += text
        if self.in_link:
            self.links[-1][1] += text


class AssembleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def document(self, body):
        return ("<!doctype html><html><body>" + assembly.SELECTOR_START
                + "<details><summary>Development</summary></details>"
                + assembly.SELECTOR_END + f"<main>{body}</main></body></html>")

    def site(self, name):
        path = self.root / name
        (path / "docs").mkdir(parents=True)
        for page in ("index.html", "docs/usage.html"):
            (path / page).write_text(self.document(f"{name}: {page}"))
        (path / "source-revision.txt").write_text(name)
        return path

    def release(self, tag="v1.0.0", published="2026-01-01T00:00:00Z", **changes):
        result = {"tag_name": tag, "published_at": published, "draft": False,
                  "prerelease": False, "assets": [{"name": "documentation.tar.gz"}]}
        result.update(changes)
        return result

    def archive(self, entries, name="documentation.tar.gz"):
        path = self.root / name
        with tarfile.open(path, "w:gz") as archive:
            for entry, content in entries:
                if isinstance(entry, str):
                    entry = tarfile.TarInfo(entry)
                data = content.encode()
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        return path

    def test_zero_one_and_multiple_snapshots(self):
        for count in (0, 1, 3):
            with self.subTest(releases=count):
                development = self.site(f"development-{count}")
                snapshots = [(f"v1.{count - i}.0", self.site(f"release-{count}-{i}"))
                             for i in range(count)]
                originals = {path: sha256(path.read_bytes()).hexdigest()
                             for directory in [development, *(p for _, p in snapshots)]
                             for path in directory.rglob("*") if path.is_file()}
                output = self.root / f"output-{count}"
                assembly.assemble(development, snapshots, output)
                labels = [tag for tag, _ in snapshots] + ["Development"]
                targets = [output / tag / "index.html" for tag, _ in snapshots]
                targets.append(output / "dev/index.html")
                current_root = labels[0] if count else "Development"
                roots = [(output, current_root), (output / "dev", "Development")]
                roots += [(output / tag, tag) for tag, _ in snapshots]
                for directory, current in roots:
                    for relative in ("index.html", "docs/usage.html"):
                        page = directory / relative
                        selector = Selector(page.read_text())
                        self.assertEqual(selector.summary.strip(), current)
                        self.assertEqual([label.strip() for _, label in selector.links], labels)
                        self.assertEqual([label.strip() for attrs, label in selector.links
                                          if attrs.get("aria-current") == "page"], [current])
                        for (attrs, _), target in zip(selector.links, targets):
                            href = attrs["href"]
                            self.assertFalse(href.startswith(("/", "http:", "https:", "file:")))
                            self.assertEqual((page.parent / href).resolve(), target.resolve())
                            self.assertTrue(target.is_file())
                expected_root = snapshots[0][1] if count else development
                self.assertEqual((output / "source-revision.txt").read_bytes(),
                                 (expected_root / "source-revision.txt").read_bytes())
                self.assertEqual({path: sha256(path.read_bytes()).hexdigest() for path in originals}, originals)

    def test_only_published_stable_snapshots_are_listed_in_publication_order(self):
        pages = [[self.release("v2.0.0"), self.release("v0.2.2", assets=[])],
                 [self.release("v1.0.0", "2026-02-01T00:00:00Z"),
                  self.release("v3.0.0", draft=True), self.release("v4.0.0", prerelease=True)]]
        self.assertEqual(assembly.snapshot_releases(pages), ["v1.0.0", "v2.0.0"])
        self.assertEqual(assembly.snapshot_releases([[self.release(assets=[])]]), [])

    def test_malformed_release_metadata_is_rejected(self):
        cases = [self.release(tag=tag) for tag in ("1.0.0", "v1.0", "../v1.0.0", "v1.0.0/extra")]
        cases += [self.release(published="yesterday"), self.release(assets={}),
                  self.release(assets=["documentation.tar.gz"]),
                  self.release(assets=[{"name": "documentation.tar.gz"}] * 2)]
        for release in cases:
            with self.subTest(release=release), self.assertRaises(ValueError):
                assembly.snapshot_releases([[release]])
        with self.assertRaises(ValueError):
            assembly.snapshot_releases([[self.release()], [self.release()]])

    def test_snapshot_extraction_preserves_files(self):
        index = self.document("release")
        archive = self.archive([("./index.html", index), ("./docs/usage.html", "guide")])
        original = sha256(archive.read_bytes()).hexdigest()
        destination = self.root / "extracted"
        assembly.extract_snapshot(archive, destination)
        self.assertEqual((destination / "index.html").read_text(), index)
        self.assertEqual((destination / "docs/usage.html").read_text(), "guide")
        self.assertEqual(sha256(archive.read_bytes()).hexdigest(), original)

    def test_unsafe_archive_entries_are_rejected(self):
        symbolic = tarfile.TarInfo("link")
        symbolic.type = tarfile.SYMTYPE
        symbolic.linkname = "index.html"
        hard = tarfile.TarInfo("hard")
        hard.type = tarfile.LNKTYPE
        hard.linkname = "index.html"
        fifo = tarfile.TarInfo("fifo")
        fifo.type = tarfile.FIFOTYPE
        cases = [[("../escaped.html", "bad")], [("/absolute.html", "bad")],
                 [("docs\\escaped.html", "bad")], [(symbolic, "")], [(hard, "")],
                 [(fifo, "")], [("./index.html", "duplicate")]]
        for i, entries in enumerate(cases):
            with self.subTest(entries=entries):
                archive = self.archive([("index.html", self.document("release")), *entries], f"bad-{i}.tar.gz")
                with self.assertRaises(ValueError):
                    assembly.extract_snapshot(archive, self.root / f"unsafe-{i}")
        self.assertFalse((self.root / "escaped.html").exists())

    def test_missing_index_or_wrong_archive_format_is_rejected(self):
        archive = self.archive([("docs/usage.html", "guide")])
        with self.assertRaises(ValueError):
            assembly.extract_snapshot(archive, self.root / "no-index")
        plain = self.root / "plain.tar"
        with tarfile.open(plain, "w") as archive:
            entry = tarfile.TarInfo("index.html")
            entry.size = 4
            archive.addfile(entry, io.BytesIO(b"html"))
        with self.assertRaises((ValueError, tarfile.ReadError)):
            assembly.extract_snapshot(plain, self.root / "not-gzip")

    def test_invalid_snapshot_inputs_are_rejected(self):
        development, release = self.site("development"), self.site("release")
        for snapshots in ([("v1.0", release)], [("v1.0.0", release)] * 2,
                          [("v1.0.0", self.root / "absent")]):
            with self.subTest(snapshots=snapshots), self.assertRaises(ValueError):
                assembly.assemble(development, snapshots, self.root / "invalid")
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises((ValueError, FileExistsError)):
            assembly.assemble(development, [], output)

    def test_legacy_snapshot_without_selector_remains_available(self):
        development, legacy = self.site("development"), self.site("legacy")
        original = b"<html>legacy documentation without a version selector</html>"
        (legacy / "index.html").write_bytes(original)
        output = self.root / "legacy-output"
        assembly.assemble(development, [("v1.0.0", legacy)], output)
        for page in (output / "index.html", output / "v1.0.0/index.html", legacy / "index.html"):
            self.assertEqual(sha256(page.read_bytes()).hexdigest(), sha256(original).hexdigest())
        selector = Selector((output / "dev/index.html").read_text())
        self.assertEqual([label.strip() for _, label in selector.links], ["v1.0.0", "Development"])
        self.assertEqual((output / "dev" / selector.links[0][0]["href"]).resolve(),
                         output / "v1.0.0/index.html")

    def test_bytes_outside_selector_are_preserved(self):
        development, release = self.site("development"), self.site("release")
        originals = {}
        for source in (development, release):
            html = (f"<!doctype html>\r\n<title>{source.name}: π</title>\r\n"
                    + assembly.SELECTOR_START + "\r\n<details>version</details>\r\n"
                    + assembly.SELECTOR_END + "\r\n<main>Zürich → λ</main>\r\n").encode("utf-8")
            (source / "index.html").write_bytes(html)
            originals[source] = html
        output = self.root / "preserved-bytes"
        assembly.assemble(development, [("v1.0.0", release)], output)

        def outside_selector(html):
            before, rest = html.split(assembly.SELECTOR_START.encode())
            _, after = rest.split(assembly.SELECTOR_END.encode())
            return before, after

        for destination, source in ((output, release), (output / "dev", development),
                                    (output / "v1.0.0", release)):
            with self.subTest(destination=destination):
                self.assertEqual(outside_selector((destination / "index.html").read_bytes()),
                                 outside_selector(originals[source]))

    def test_partial_or_duplicate_selector_is_rejected(self):
        for i, malformed in enumerate((assembly.SELECTOR_START, assembly.SELECTOR_END,
                                       assembly.SELECTOR_START * 2 + assembly.SELECTOR_END)):
            development = self.site(f"development-{i}")
            (development / "docs/usage.html").write_text(malformed)
            with self.subTest(marker=malformed), self.assertRaises(ValueError):
                assembly.assemble(development, [], self.root / f"malformed-selector-{i}")


if __name__ == "__main__":
    unittest.main()
