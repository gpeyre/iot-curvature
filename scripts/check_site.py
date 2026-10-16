"""Check the self-contained GitHub Pages site without making network requests."""
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "docs").resolve()
errors = []
local_references = 0


def label(path):
    return str(path.relative_to(ROOT))


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.ids = set()
        self.references = []
        self.stylesheets = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            element_id = attrs["id"]
            if element_id in self.ids:
                errors.append(f"{label(self.path)}: duplicate id {element_id!r}")
            self.ids.add(element_id)
        for attribute in ("href", "src"):
            if attrs.get(attribute) is not None:
                self.references.append((attrs[attribute], self.getpos()[0]))
        if tag == "link" and "stylesheet" in (attrs.get("rel") or "").split():
            self.stylesheets.append(attrs.get("href"))

    handle_startendtag = handle_starttag


def check_reference(origin, reference, location=None):
    """Resolve a site URL while enforcing GitHub project-subpath portability."""
    global local_references
    context = f"{label(origin)}:{location}" if location else label(origin)
    try:
        url = urlsplit(reference.strip())
    except ValueError as exc:
        errors.append(f"{context}: invalid URL {reference!r}: {exc}")
        return None
    if url.scheme or url.netloc:
        return None  # External links, mailto:, data:, node:, etc. are not fetched.
    path = unquote(url.path)
    if path.startswith("/") or "\\" in path:
        errors.append(f"{context}: use a relative project-site URL: {reference!r}")
        return None
    target = (origin.parent / path).resolve() if path else origin
    if not target.is_relative_to(DOCS):
        errors.append(f"{context}: URL escapes docs/: {reference!r}")
        return None
    if target.is_dir():
        target /= "index.html"
    local_references += 1
    if not target.is_file():
        errors.append(f"{context}: missing local target {reference!r}")
        return None
    return target, unquote(url.fragment)


pages = {}
for path in sorted(DOCS.rglob("*.html")):
    page = Page(path)
    page.feed(path.read_text(encoding="utf-8"))
    page.close()
    pages[path] = page

for required in ("index.html", "demo.html"):
    if DOCS / required not in pages:
        errors.append(f"docs/{required}: required site page is missing")
    elif not pages[DOCS / required].stylesheets:
        errors.append(f"docs/{required}: page has no linked stylesheet")

for path, page in pages.items():
    for reference, line in page.references:
        resolved = check_reference(path, reference, line)
        if resolved:
            target, fragment = resolved
            if fragment and target in pages and fragment not in pages[target].ids:
                errors.append(f"{label(path)}:{line}: missing fragment in {reference!r}")

css_url = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
for path in sorted(DOCS.rglob("*.css")):
    source = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    for match in css_url.finditer(source):
        check_reference(path, match.group(2))

# The site uses static ES imports and module-relative worker URLs. These narrow
# patterns deliberately avoid pretending to parse arbitrary JavaScript.
js_dependencies = re.compile(
    r"\bimport\s+(?:[\w*$,\s{}]+\s+from\s+)?['\"]([^'\"]+)['\"]"
    r"|\bnew\s+URL\(\s*['\"]([^'\"]+)['\"]\s*,\s*import\.meta\.url\s*\)"
)
for path in sorted(DOCS.glob("*.mjs")):
    for match in js_dependencies.finditer(path.read_text(encoding="utf-8")):
        check_reference(path, match.group(1) or match.group(2))

figures = sorted((ROOT / "figures" / "generated").glob("figure_[1-4]_*.png"))
if len(figures) != 4:
    errors.append(f"Expected four source figure PNGs, found {len(figures)}")
for original in figures:
    copied = DOCS / "assets" / original.name
    if not copied.is_file():
        errors.append(f"{label(copied)}: copied publication preview is missing")
    elif sha256(original.read_bytes()).digest() != sha256(copied.read_bytes()).digest():
        errors.append(f"{label(copied)}: differs from {label(original)}; run prepare_site.py")
if not (DOCS / "assets" / "experiment-preview.svg").is_file():
    errors.append("docs/assets/experiment-preview.svg: demo preview is missing")

if errors:
    print("Static-site checks failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"Static site OK: {len(pages)} pages, {local_references} local references, "
      f"4 matching publication previews; all URLs support a project subpath.")
