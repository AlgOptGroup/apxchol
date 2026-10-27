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
release_tag = os.environ.get("APXCHOL_DOCS_RELEASE", "")
if release_tag:
    if release_tag != f"v{release}":
        raise ValueError(f"Documentation tag {release_tag!r} does not match v{release}")
html_title = "apxchol"
html_context = {"docs_version": release_tag or "Development"}
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["versions.css"]
html_sidebars = {"**": [
    "sidebar/brand.html", "sidebar/version.html", "sidebar/search.html",
    "sidebar/scroll-start.html", "sidebar/navigation.html", "sidebar/ethical-ads.html",
    "sidebar/scroll-end.html", "sidebar/variant-selector.html",
]}
html_show_sourcelink = False
html_show_copyright = False
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 4
breathe_projects = {"apxchol": str(Path(os.environ["APXCHOL_DOCS_BUILD"]) / "doxygen/xml")}
breathe_default_project = "apxchol"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
