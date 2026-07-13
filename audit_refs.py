#!/usr/bin/env python3
"""Reference auditor — scans a .tex file, flags suspicious references."""
import re
import sys
import subprocess
from pathlib import Path

# Known real journals (verified in previous audits)
VERIFIED_JOURNAL_HINTS = [
    ("AJ", "The Astronomical Journal"),
    ("ApJ", "The Astrophysical Journal"),
    ("ApJL", "Astrophysical Journal Letters"),
    ("ApJS", "Astrophysical Journal Supplement"),
    ("MNRAS", "Monthly Notices of the RAS"),
    ("Nature", "Nature"),
    ("Nature Astronomy", "Nature Astronomy"),
    ("Icarus", "Icarus"),
    ("PASP", "Publ. Astron. Soc. Pacific"),
    ("A&A", "Astronomy & Astrophysics"),
    ("PSJ", "The Planetary Science Journal"),
    ("Planet. Sci. J.", "The Planetary Science Journal"),
    ("Planetary Science Journal", "The Planetary Science Journal"),
    ("JATIS", "J. Astron. Telesc. Instrum. Syst."),
    ("arXiv", "arXiv preprint"),
    ("PNAS", "Proc. Natl. Acad. Sci."),
    ("PRX", "Physical Review X"),
    ("PRD", "Physical Review D"),
    ("CVPR", "CVPR Conference"),
    ("ICML", "ICML Conference"),
    ("MNRAS", "Monthly Notices"),
    ("Wiley", "Wiley"),
    ("Chapman", "Chapman & Hall"),
    ("Cambridge", "Cambridge University Press"),
    ("University of Arizona Press", "Book"),
    ("IEEE Trans", "IEEE Transactions"),
    ("Journal of Computational Physics", "J. Comp. Phys."),
    ("JQSRT", "J. Quant. Spectrosc. Radiat. Transfer"),
    ("MICCAI", "MICCAI Conference"),
]


def extract_refs_from_tex(tex_path):
    """Extract references from .tex file."""
    text = Path(tex_path).read_text(encoding='utf-8', errors='replace')
    body, _, bib = text.partition(r'\begin{thebibliography}')
    if not bib:
        body, _, bib = text.partition(r'\begin{thebibliography*}')
    
    refs = []
    # Author-year format: \bibitem[Author(Year)]{key}
    for m in re.finditer(r'\\bibitem\[([^\]]*)\]\{([^}]+)\}', bib):
        label = m.group(1).strip()
        key = m.group(2).strip()
        # Get text until next bibitem
        start = m.end()
        end_match = re.search(r'\\bibitem', bib[start:])
        body_text = bib[start:start+end_match.start()] if end_match else bib[start:]
        body_text = body_text.strip()
        refs.append({"key": key, "label": label, "body": body_text[:300]})
    
    # Simple format: \bibitem{key}
    for m in re.finditer(r'(?<!\[)\\\bibitem\{([^}]+)\}', bib):
        if m.group(1) not in [r['key'] for r in refs]:
            key = m.group(1).strip()
            start = m.end()
            end_match = re.search(r'\\bibitem', bib[start:])
            body_text = bib[start:start+end_match.start()] if end_match else bib[start:]

            refs.append({"key": key, "label": key, "body": body_text.strip()[:300]})

    return body, refs


def verify_online(author, year, title_hint, journal_hint):
    """Attempt a quick web search to verify the reference exists."""
    query = f'{author} {year} "{title_hint}" {journal_hint}'
    try:
        # Try simple retrieval — will fail gracefully if offline
        result = subprocess.run(
            ['python3', '-c', f'''
import requests
try:
    r = requests.get("https://api.openalex.org/works", params={{"search": "{query[:200]}"}}, timeout=5)
    if r.status_code == 200:
        data = r.json()
        count = data.get("meta", {{}}).get("count", 0)
        print(count)
    else:
        print("API_ERROR")
except:
    print("OFFLINE")
'''],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        if output.isdigit():
            return int(output) > 0
        return None  # Could not verify (offline or API error)
    except:
        return None


def audit(tex_path):
    """Main audit function — returns a report."""
    body, refs = extract_refs_from_tex(tex_path)

    report = []
    flags = 0

    # Check each reference has a citation in the body
    for r in refs:
        key = r['key']
        cited = key in body
        if not cited:
            r['uncited'] = True
            flags += 1
        else:
            r['uncited'] = False

    # Check for suspicious patterns
    for r in refs:
        body_text = r['body']

        # Flag #1: Future year?
        year_match = re.search(r'\b(20[2-9]\d)\b', body_text)
        if year_match:
            year = int(year_match.group(1))
            if year > 2026:
                r['future_year'] = True
                flags += 1
            else:
                r['future_year'] = False
        else:
            r['future_year'] = None

        # Flag #2: "submitted" or "in press" without journal?
        if re.search(r'submitted|in press|to appear', body_text, re.IGNORECASE):
            if not re.search(r'arXiv', body_text, re.IGNORECASE):
                r['no_preprint'] = True
                flags += 1
            else:
                r['no_preprint'] = False
        else:
            r['no_preprint'] = False

        # Flag #3: DOI present?
        r['has_doi'] = bool(re.search(r'\b10\.\d{4,}/', body_text))

        # Flag #4: Known journal?
        journal_known = False
        for hint, _ in VERIFIED_JOURNAL_HINTS:
            if hint.lower() in body_text.lower():
                journal_known = True
                break
        r['journal_known'] = journal_known
        if not journal_known and 'arXiv' not in body_text:
            r['unknown_journal'] = True
            flags += 1
        else:
            r['unknown_journal'] = False

        # Extract author name for search
        author_match = re.search(r'([A-Z][a-z]+(?:[- ][A-Z][a-z]+)?)', body_text)
        r['first_author_hint'] = author_match.group(1) if author_match else '?'

    report = sorted(refs, key=lambda r: (
        int(r.get('uncited', False)),
        int(r.get('future_year', False) or False),
        int(r.get('no_preprint', False) or False),
        int(r.get('unknown_journal', False) or False),
    ), reverse=True)

    return body, report, flags


def print_report(body, report, flags, tex_path):
    """Print a human-readable audit report."""
    print(f"Reference audit: {tex_path}")
    print(f"Total references: {len(report)}")
    print(f"Flags raised: {flags}")
    print()

    if flags == 0:
        print("✅ All references appear well-formed.")
        print("   (Verification of content requires manual ADS/arXiv lookup.)")
        print()
        for r in report:
            print(f"  ✅ {r['key']}: {r['body'][:100]}...")
    else:
        for r in report:
            issues = []
            if r.get('uncited'):
                issues.append('❌ UNCITED')
            if r.get('future_year'):
                issues.append('⚠️ FUTURE YEAR')
            if r.get('no_preprint'):
                issues.append('⚠️ NO PREPRINT (submitted/in press)')
            if r.get('unknown_journal'):
                issues.append('⚠️ UNKNOWN JOURNAL')
            if r.get('has_doi'):
                issues.append('✅ DOI present')
            if not issues:
                issues.append('✅ Clean')

            status = ' | '.join(issues)
            print(f"  {status}")
            print(f"    Key: {r['key']}")
            print(f"    {r['body'][:150]}...")
            print()

    # Print ADS search URLs for manual verification
    print("=" * 60)
    print("Manual verification URLs (paste into ADS: ui.adsabs.harvard.edu):")
    print("=" * 60)
    for r in report:
        author = r.get('first_author_hint', '')
        year_match = re.search(r'\b(19|20\d{2})\b', r['body'])
        year = year_match.group(1) if year_match else ''
        print(f"  {author} {year} — {r['key']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 audit_refs.py <file.tex>")
        sys.exit(1)

    tex_path = sys.argv[1]
    body, report, flags = audit(tex_path)
    print_report(body, report, flags, tex_path)
