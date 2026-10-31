"""Check the citing page against CITATION.cff."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CitingTest(unittest.TestCase):
    def test_software_entry_matches_citation_file(self):
        cff = (ROOT / "CITATION.cff").read_text()
        families = re.findall(r"family-names: (.+)", cff)
        givens = re.findall(r"given-names: (.+)", cff)
        title = re.search(r'^title: "(.+)"$', cff, re.M).group(1)
        page = (ROOT / "docs/citing.md").read_text()
        authors = " and ".join(f"{f}, {g}" for f, g in zip(families, givens))
        self.assertIn(f"author  = {{{authors}}}", page)
        self.assertIn(title.replace("apxchol", "{apxchol}", 1), page)
        self.assertIn(re.search(r'repository-code: "(.+)"', cff).group(1), page)
        self.assertIn(re.search(r"license: (.+)", cff).group(1), page)


if __name__ == "__main__":
    unittest.main()
