"""Nav sync utility — keeps the nav bar identical across all pages.

Usage:
    uv run python src/build_site.py          # Sync nav from index.html to all pages
    uv run python src/build_site.py --check  # Dry run, show what would change

Each page in docs/ is standalone HTML (no build step needed for content).
This script simply propagates nav bar changes so you don't have to
copy-paste the same nav edit into every file.
"""

import re
from pathlib import Path

DOCS = Path("docs")
NAV_PATTERN = re.compile(
    r'(<nav class="site-nav">.*?</nav>)',
    re.DOTALL,
)
SKIP = {"index.html"}


def get_nav_block(path):
    m = NAV_PATTERN.search(path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def get_slug(path):
    return path.stem  # "index", "shot", etc.


def set_active_link(nav_html, slug):
    """Set the 'active' class only on the nav link for slug."""
    def replace_link(m):
        href = m.group(1)
        cls = m.group(2) or ""
        rest = m.group(3)
        is_target = href == f"./{slug}.html"
        classes = [c for c in cls.split() if c != "active"]
        if is_target:
            classes.append("active")
        new_cls = (" " + " ".join(classes)).rstrip()
        return f'<a href="{href}" class="{new_cls}"{rest}' if new_cls.strip() else f'<a href="{href}"{rest}'

    return re.sub(
        r'<a href="(\./[^"]+)"\s*class="([^"]*)"([^>]*>)',
        replace_link,
        nav_html,
    )


def sync(check_only=False):
    ref = DOCS / "index.html"
    ref_nav = get_nav_block(ref)
    if not ref_nav:
        print("ERROR: Could not find nav block in index.html")
        return

    changed = []
    for f in sorted(DOCS.glob("*.html")):
        if f.name in SKIP:
            continue
        slug = get_slug(f)
        old = f.read_text(encoding="utf-8")
        current_nav = get_nav_block(f)
        if not current_nav:
            print(f"  SKIP {f.name} — no nav block found")
            continue

        new_nav = set_active_link(ref_nav, slug)

        if current_nav == new_nav:
            continue

        new_html = old.replace(current_nav, new_nav)
        if not check_only:
            f.write_text(new_html, encoding="utf-8")
        changed.append(f.name)

    if changed:
        print(f"  {'[DRY RUN] ' if check_only else ''}Updated nav in: {', '.join(changed)}")
    else:
        print("  All nav blocks already in sync.")


if __name__ == "__main__":
    import sys
    sync(check_only="--check" in sys.argv)
