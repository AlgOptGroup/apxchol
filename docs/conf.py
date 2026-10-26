import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
project = "apxchol"
author = "The apxchol authors"
release = re.search(r"APXCHOL_VERSION ([0-9.]+)",
                    (ROOT / "cmake/apxchol_version.cmake").read_text()).group(1)
version = release
extensions = ["myst_parser", "breathe", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_readme.md"]
html_theme = "furo"
html_theme_options = {
    "announcement": 'Development documentation from <code>main</code>. '
                    'See <a href="https://pypi.org/project/apxchol/">PyPI</a> for the released package.'
}
html_title = f"apxchol {release}"
html_show_sourcelink = False
html_show_copyright = False
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 4
breathe_projects = {"apxchol": str(Path(os.environ["APXCHOL_DOCS_BUILD"]) / "doxygen/xml")}
breathe_default_project = "apxchol"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
