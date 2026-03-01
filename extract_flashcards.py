#!/usr/bin/env python3
"""
Portuguese Lesson → Quizlet Flashcard Extractor

Extracts vocabulary words, phrases, and key expressions from Portuguese
lessons (PDF files or Google Docs URLs) and outputs them in Quizlet's
import format.

Usage:
    python3 extract_flashcards.py <source> [--output <output_path>]

    # From a PDF file:
    python3 extract_flashcards.py lesson.pdf

    # From a Google Doc (must be shared with "Anyone with the link"):
    python3 extract_flashcards.py "https://docs.google.com/document/d/1abc.../edit"

    python3 extract_flashcards.py lesson.pdf --output my_flashcards.txt

The output file can be pasted directly into Quizlet's "Import" feature.
"""

import re
import sys
import argparse
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Section filtering – skip exercises, homework, practice sections
# ---------------------------------------------------------------------------

# Headings that signal an EXERCISE section (not flashcard material)
EXERCISE_HEADERS = re.compile(
    r"(?i)"
    r"(sua vez|exerc[ií]cio|fill in|multiple choice|true or false|"
    r"verdadeiro ou falso|translate into|traduza|match with|"
    r"choose the|escolha|complete (with|com)|comprehension|"
    r"perguntas de compreens|role-play|visual activity|"
    r"visual shopping|responda oralmente|short answer|"
    r"create a sentence|produção própria|dialogue keyword|"
    r"transform into|gap-fill|exercício visual|"
    r"homework|fill in the blank|choose the right)"
)

# Headings that signal we're back in CONTENT territory
CONTENT_HEADERS = re.compile(
    r"(?i)"
    r"(vocabulary|vocabul[aá]rio|introduction|introdução|"
    r"phrases for|key expression|preposições de|"
    r"preposição |contração|tabela de|"
    r"dialogue\b|períodos|time expression|"
    r"expressões comuns|lesson \d|parte de cima|"
    r"parte de baixo|roupas especiais|acessórios de|"
    r"calçados|outros itens|exemplos\b|examples\b|"
    r"ser de|morar em|articles and|cidades|"
    r"numbers for|the usage of|dizendo a hora|"
    r"meia hora|quinze minutos|grave accent|"
    r"important notes|tips\b|conclusion|"
    r"visual table)"
)


def is_exercise_heading(text: str) -> bool:
    return bool(EXERCISE_HEADERS.search(text))


def is_content_heading(text: str) -> bool:
    return bool(CONTENT_HEADERS.search(text))


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Strip whitespace, emojis, and trailing/leading punctuation noise."""
    if not text:
        return ""
    # Remove common emoji/icon characters (broad Unicode ranges)
    text = re.sub(r'[\U0001F300-\U0001FAFF\U000F0000-\U000FFFFF\u2600-\u27BF\u2700-\u27BF\u2B50\u2705\u274C\u2714\u2716✅❌✏️📖📚🎒👜👗👕👖🧣🧤👙👢👡👟👞🧢👒🎩💍📿👓🕶⌚💎🌂☂️🎓■]', '', text)
    text = text.strip(" \t\n•●○◦▪▸►→←↔·")
    # Collapse internal multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_arrow(text: str) -> str:
    """Normalize different arrow characters to →."""
    return re.sub(r'\s*[→⟶⇒]\s*', ' → ', text)


def extract_arrow_pairs(text: str) -> list[tuple[str, str]]:
    """
    Extract pairs from text containing → arrows.
    Handles both:
      - "word → translation"  (single words)
      - "Phrase in Portuguese. → English translation."  (full phrases)
    Also handles two-column layouts where pdfplumber merges lines with ●.
    """
    pairs = []
    lines = text.split('\n')
    for line in lines:
        line = normalize_arrow(line)

        # Handle two-column merges: "em → in / at / on ● entre → between"
        # Split on ● and process each part independently
        parts = re.split(r'\s*[●•]\s*', line)

        for part in parts:
            # Match: something → something
            m = re.search(r'(.+?)\s*→\s*(.+)', part)
            if m:
                left = clean(m.group(1))
                right = clean(m.group(2))
                if not left or not right:
                    continue
                # Skip exercise arrows (lines with [ ] blanks or ______)
                if '[ ]' in left or '[ ]' in right:
                    continue
                if '______' in left or '______' in right:
                    continue
                # Skip lines that are just section labels
                if left.lower() in ('lugar', 'tempo', 'cities'):
                    continue
                # Skip summary notes (not actual flashcard material)
                if 'almost always' in right.lower():
                    continue
                pairs.append((left, right))
    return pairs


def extract_table_pairs(table: list[list[str]]) -> list[tuple[str, str]]:
    """
    Extract Portuguese/English pairs from a table.
    Handles tables with headers like:
      - Portuguese | English
      - Preposição | Uso | Contrações | Exemplo | Tradução
      - Forma em Português | Exemplo | Tradução
    """
    if not table or len(table) < 2:
        return []

    pairs = []
    headers = [clean(str(h)).lower() if h else "" for h in table[0]]

    # Strategy 1: Simple 2-column Portuguese/English table
    pt_col = None
    en_col = None
    for i, h in enumerate(headers):
        if any(k in h for k in ('portugu', 'preposição', 'forma')):
            pt_col = i
        if any(k in h for k in ('english', 'translat', 'tradução')):
            en_col = i

    if pt_col is not None and en_col is not None:
        for row in table[1:]:
            if row and len(row) > max(pt_col, en_col):
                pt = clean(str(row[pt_col])) if row[pt_col] else ""
                en = clean(str(row[en_col])) if row[en_col] else ""
                if pt and en and '[ ]' not in pt and '[ ]' not in en:
                    pairs.append((pt, en))
        return pairs

    # Strategy 2: Preposição table with Exemplo + Tradução columns
    prep_col = None
    uso_col = None
    exemplo_col = None
    trad_col = None
    for i, h in enumerate(headers):
        if 'preposição' in h or 'forma' in h:
            prep_col = i
        if 'uso' in h:
            uso_col = i
        if 'exemplo' in h:
            exemplo_col = i
        if 'tradução' in h or 'translat' in h:
            trad_col = i

    if prep_col is not None and uso_col is not None:
        for row in table[1:]:
            if not row or len(row) <= max(prep_col, uso_col):
                continue
            prep = clean(str(row[prep_col])) if row[prep_col] else ""
            uso = clean(str(row[uso_col])) if row[uso_col] else ""
            if prep and uso:
                # Card: preposition → usage/meaning
                # Extract just the English meaning in parentheses if present
                m = re.search(r'\("([^"]+)"\)', uso)
                if m:
                    pairs.append((prep, m.group(1)))
                else:
                    pairs.append((prep, uso))
            # Also add exemplo → tradução if available
            if exemplo_col is not None and trad_col is not None:
                if len(row) > max(exemplo_col, trad_col):
                    ex = clean(str(row[exemplo_col])) if row[exemplo_col] else ""
                    tr = clean(str(row[trad_col])) if row[trad_col] else ""
                    if ex and tr:
                        # Split multi-sentence examples (separated by " / ")
                        ex_parts = ex.split(' / ')
                        tr_parts = tr.split(' / ')
                        if len(ex_parts) == len(tr_parts) and len(ex_parts) > 1:
                            for e, t in zip(ex_parts, tr_parts):
                                e, t = clean(e), clean(t)
                                if e and t:
                                    pairs.append((e, t))
                        else:
                            pairs.append((ex, tr))
        return pairs

    # Strategy 3: Table with just Exemplo + Tradução
    if exemplo_col is not None and trad_col is not None:
        for row in table[1:]:
            if row and len(row) > max(exemplo_col, trad_col):
                ex = clean(str(row[exemplo_col])) if row[exemplo_col] else ""
                tr = clean(str(row[trad_col])) if row[trad_col] else ""
                if ex and tr and '[ ]' not in ex and '[ ]' not in tr:
                    pairs.append((ex, tr))
        return pairs

    return pairs


def extract_parenthetical_translations(text: str) -> list[tuple[str, str]]:
    """
    Extract phrases where the pattern is:
      Portuguese sentence. (English translation)
    Common in dialogue and example sections.
    """
    pairs = []
    # Pattern: "Portuguese text. (English text)" or "(English text)"
    # Look for lines like: "O livro está no sofá. (The book is on the sofa)"
    pattern = re.compile(r'([A-ZÀ-Ú][^()\n]{5,}?[.!?])\s*\(([A-Z][^()]+)\)')
    for m in pattern.finditer(text):
        pt = clean(m.group(1))
        en = clean(m.group(2))
        if pt and en and '[ ]' not in pt:
            pairs.append((pt, en))
    return pairs


# ---------------------------------------------------------------------------
# Google Docs support
# ---------------------------------------------------------------------------

GDOC_URL_PATTERN = re.compile(
    r'https?://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)'
)


def is_google_doc_url(source: str) -> bool:
    """Check if the source is a Google Docs URL."""
    return bool(GDOC_URL_PATTERN.match(source))


def extract_doc_id(url: str) -> str:
    """Extract the document ID from a Google Docs URL."""
    m = GDOC_URL_PATTERN.match(url)
    if not m:
        raise ValueError(f"Could not extract document ID from URL: {url}")
    return m.group(1)


def fetch_google_doc_html(doc_id: str) -> str:
    """
    Fetch a Google Doc as HTML using the export endpoint.
    Works for docs shared with "Anyone with the link can view".
    """
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"
    resp = requests.get(export_url, timeout=30)
    if resp.status_code == 404:
        raise ValueError(
            "Document not found. Make sure the URL is correct and the doc is "
            "shared with 'Anyone with the link'."
        )
    if resp.status_code != 200:
        raise ValueError(
            f"Failed to fetch document (HTTP {resp.status_code}). "
            "Make sure the doc is shared with 'Anyone with the link'."
        )
    return resp.text


def extract_flashcards_from_html(html: str) -> list[tuple[str, str]]:
    """
    Extract flashcard pairs from Google Docs HTML export.
    Uses the same quality filters as the PDF extractor but with
    cleaner input (no smashed words or merged columns).
    """
    soup = BeautifulSoup(html, 'html.parser')
    all_pairs = []
    seen = set()

    def add_pair(pt: str, en: str):
        pt = re.sub(r'\s+', ' ', pt).strip()
        en = re.sub(r'\s+', ' ', en).strip()
        key = (pt.lower(), en.lower())
        if key in seen or not pt or not en:
            return
        if '______' in pt or '______' in en:
            return
        if '[ ]' in pt or '[ ]' in en:
            return
        if len(pt) > 200 or len(en) > 200:
            return
        # Skip truncated translations
        if len(pt) > 20 and len(en.split()) <= 2 and not en.endswith('.'):
            return
        # Skip fragment endings (only for phrases, not single-word translations like "em → in")
        if re.search(r'\b(in|of|the|a|an|to|for|at|is|it|from|with)\s*$', en.lower()) and len(en) < 30 and len(pt.split()) > 1:
            return
        seen.add(key)
        all_pairs.append((pt, en))

    in_exercise_section = False

    # --- Extract from HTML tables ---
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue

        # Parse header row
        header_cells = rows[0].find_all(['td', 'th'])
        headers = [clean(cell.get_text()) .lower() for cell in header_cells]

        # Check if this looks like an exercise table
        table_text = table.get_text()
        if '[ ]' in table_text or '______' in table_text:
            blank_count = table_text.count('[ ]') + table_text.count('______')
            if blank_count > 2:
                continue

        # Find relevant columns
        pt_col = en_col = exemplo_col = trad_col = uso_col = None
        for i, h in enumerate(headers):
            if any(k in h for k in ('portugu', 'preposição', 'forma')):
                pt_col = i
            if any(k in h for k in ('english', 'translat', 'tradução')):
                en_col = i
            if 'exemplo' in h:
                exemplo_col = i
            if 'tradução' in h or 'translat' in h:
                trad_col = i
            if 'uso' in h:
                uso_col = i

        # Strategy 1: Preposição + Uso + Exemplo + Tradução
        # Check BEFORE simple Portuguese/English — preposição tables also
        # have tradução which would falsely match as a simple en_col
        if (pt_col is not None or any('preposição' in h for h in headers)) and uso_col is not None:
            prep_col = pt_col if pt_col is not None else next(
                (i for i, h in enumerate(headers) if 'preposição' in h), None
            )
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if prep_col is not None and len(cells) > max(prep_col, uso_col):
                    prep = clean(cells[prep_col].get_text())
                    uso = clean(cells[uso_col].get_text())
                    if prep and uso:
                        m = re.search(r'\("([^"]+)"\)', uso)
                        if m:
                            add_pair(prep, m.group(1))
                        else:
                            add_pair(prep, uso)
                if exemplo_col is not None and trad_col is not None:
                    if len(cells) > max(exemplo_col, trad_col):
                        ex = clean(cells[exemplo_col].get_text())
                        tr = clean(cells[trad_col].get_text())
                        if ex and tr:
                            ex_parts = ex.split(' / ')
                            tr_parts = tr.split(' / ')
                            if len(ex_parts) == len(tr_parts) and len(ex_parts) > 1:
                                for e, t in zip(ex_parts, tr_parts):
                                    e, t = clean(e), clean(t)
                                    if e and t:
                                        add_pair(e, t)
                            else:
                                add_pair(ex, tr)
            continue

        # Strategy 2: Simple Portuguese / English columns
        if pt_col is not None and en_col is not None:
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) > max(pt_col, en_col):
                    pt = clean(cells[pt_col].get_text())
                    en = clean(cells[en_col].get_text())
                    if pt and en:
                        add_pair(pt, en)
            continue

        # Strategy 3: Exemplo + Tradução only
        if exemplo_col is not None and trad_col is not None:
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) > max(exemplo_col, trad_col):
                    ex = clean(cells[exemplo_col].get_text())
                    tr = clean(cells[trad_col].get_text())
                    if ex and tr:
                        add_pair(ex, tr)
            continue

    # --- Extract from text content (arrow pairs, parenthetical translations) ---
    # Get all text, processing element by element for section awareness
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span']):
        # Skip elements inside tables (already processed)
        if element.find_parent('table'):
            continue

        text = element.get_text().strip()
        if not text:
            continue

        # Track exercise vs content sections
        if is_exercise_heading(text):
            in_exercise_section = True
            continue
        if is_content_heading(text):
            in_exercise_section = False
            continue
        if re.match(r'.*Lesson \d+', text):
            in_exercise_section = False
            continue

        if in_exercise_section:
            continue

        # Arrow pairs
        for pt, en in extract_arrow_pairs(text):
            add_pair(pt, en)

        # Parenthetical translations
        for pt, en in extract_parenthetical_translations(text):
            add_pair(pt, en)

    return all_pairs


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_flashcards(pdf_path: str) -> list[tuple[str, str]]:
    """
    Extract all flashcard pairs from a Portuguese lesson PDF.
    Returns list of (portuguese, english) tuples.
    """
    all_pairs = []
    seen = set()  # Dedup

    def has_smashed_words(text: str) -> bool:
        """Detect text where words got smashed together by PDF extraction."""
        # Split on spaces and check each token
        tokens = text.split()
        for token in tokens:
            # Check each slash-separated part
            parts = token.split('/') if '/' in token else [token]
            for p in parts:
                p_stripped = p.strip('().,!?')
                if not p_stripped:
                    continue
                # CamelCase-like: lowercase→uppercase in the middle of a word
                if re.search(r'[a-zà-ú][A-ZÀ-Ú]', p_stripped):
                    return True
                # Accented vowel followed by consonant start of new word (e.g., "Estácalor")
                if re.search(r'[áéíóúà][a-z]{2,}', p_stripped) and len(p_stripped) > 6:
                    return True
                # Common Portuguese verb smashing: "Faz" + word, "Está" + word with no space
                if re.match(r'^(Faz|Tá)[a-z]', p_stripped) and len(p_stripped) > 5:
                    return True
                # Table formatting artifacts with "+" signs (e.g., "Às+hora")
                if '+' in p_stripped and len(p_stripped) > 4:
                    return True
                # Words that are suspiciously long with no hyphens (e.g., "Comminutos", "Quinzepara")
                if len(p_stripped) > 11 and '-' not in p_stripped and '(' not in p_stripped:
                    # Allow known long Portuguese/English words
                    if not re.match(r'^[A-Za-zÀ-ú]+mente$', p_stripped):  # -mente adverbs OK
                        if not re.match(r'^(experimentar|temperatura|supermercado|interesting)$', p_stripped, re.I):
                            return True
        return False

    def add_pair(pt: str, en: str):
        # Clean multi-line text from table cells
        pt = re.sub(r'\s+', ' ', pt).strip()
        en = re.sub(r'\s+', ' ', en).strip()
        key = (pt.lower(), en.lower())
        if key in seen or not pt or not en:
            return
        # Filter out garbage entries
        if '______' in pt or '______' in en:
            return
        if '[ ]' in pt or '[ ]' in en:
            return
        # Skip entries that are clearly not flashcard material
        if len(pt) > 200 or len(en) > 200:
            return
        # Skip entries with smashed-together text from bad PDF table extraction
        if has_smashed_words(pt) or has_smashed_words(en):
            return
        # Skip truncated translations (ending with a preposition or just 1-2 words for a long phrase)
        if len(pt) > 20 and len(en.split()) <= 2 and not en.endswith('.'):
            return
        # Skip if English side is clearly a fragment (only for phrases, not single-word translations)
        if re.search(r'\b(in|of|the|a|an|to|for|at|is|it|from|with)\s*$', en.lower()) and len(en) < 30 and len(pt.split()) > 1:
            return
        # Skip merged adverb+sentence table entries (e.g., "sempre Eu estudo português.")
        # Pattern: Portuguese adverb followed by a capital letter starting a new sentence
        if re.match(r'^(sempre|geralmente|normalmente|frequentemente|raramente|nunca|muito|pouco|bastante|demais|àsvezes)\s+[A-ZÀ-Ú]', pt):
            return
        if re.match(r'^(always|usually|normally|sometimes|rarely|never|very|alittle|alot|extremely|too)\s+[A-ZÀ-Ú]', en):
            return
        seen.add(key)
        all_pairs.append((pt, en))

    with pdfplumber.open(pdf_path) as pdf:
        in_exercise_section = False

        for page in pdf.pages:
            text = page.extract_text() or ""

            # --- Process tables on this page ---
            tables = page.extract_tables() or []
            for table in tables:
                if not table:
                    continue
                # Check if first row looks like an exercise (Sua vez)
                first_cell = str(table[0][0]).lower() if table[0] and table[0][0] else ""
                if '[ ]' in str(table) and any('[ ]' in str(cell) for row in table for cell in row if cell):
                    # Check if it's a "Sua vez" practice table (lots of blanks)
                    blank_count = sum(1 for row in table for cell in row if cell and '[ ]' in str(cell))
                    total_cells = sum(1 for row in table for cell in row if cell)
                    if total_cells > 0 and blank_count / total_cells > 0.3:
                        continue  # Skip exercise tables

                for pt, en in extract_table_pairs(table):
                    add_pair(pt, en)

            # --- Process text line by line with section awareness ---
            lines = text.split('\n')
            for line in lines:
                stripped = line.strip()

                # Track whether we're in an exercise section
                if is_exercise_heading(stripped):
                    in_exercise_section = True
                    continue
                if is_content_heading(stripped):
                    in_exercise_section = False
                    continue
                # New lesson header resets
                if re.match(r'.*Lesson \d+', stripped):
                    in_exercise_section = False
                    continue

                if in_exercise_section:
                    continue

                # Extract arrow pairs from this line
                for pt, en in extract_arrow_pairs(line):
                    add_pair(pt, en)

            # --- Extract parenthetical translations (not in exercises) ---
            if not in_exercise_section:
                for pt, en in extract_parenthetical_translations(text):
                    add_pair(pt, en)

    return all_pairs


def format_for_quizlet(pairs: list[tuple[str, str]], delimiter: str = '\t') -> str:
    """
    Format pairs for Quizlet import.
    Default: tab-separated, one card per line.
    """
    lines = []
    for pt, en in pairs:
        # Clean any remaining tabs/newlines from within the values
        pt_clean = pt.replace('\t', ' ').replace('\n', ' ')
        en_clean = en.replace('\t', ' ').replace('\n', ' ')
        lines.append(f"{pt_clean}{delimiter}{en_clean}")
    return '\n'.join(lines)


def group_by_lesson(pairs: list[tuple[str, str]], pdf_path: str) -> dict[str, list[tuple[str, str]]]:
    """
    Attempt to group flashcards by lesson number.
    Returns dict mapping lesson name to pairs.
    Currently returns all under one group since extraction is page-based.
    """
    # For now, return all as one group. Future enhancement: track page->lesson mapping.
    name = Path(pdf_path).stem
    return {name: pairs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Portuguese flashcards from lesson PDFs or Google Docs for Quizlet import"
    )
    parser.add_argument(
        "source",
        help="Path to a lesson PDF, or a Google Docs URL (shared with 'Anyone with the link')"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: flashcards.txt for URLs, or same name as PDF with .txt)"
    )
    parser.add_argument(
        "--delimiter", "-d",
        default="\t",
        help="Delimiter between term and definition (default: tab)"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview extracted flashcards in the terminal"
    )
    args = parser.parse_args()

    source = args.source

    if is_google_doc_url(source):
        # --- Google Docs mode ---
        doc_id = extract_doc_id(source)
        print(f"Fetching Google Doc: {doc_id[:20]}...")
        try:
            html = fetch_google_doc_html(doc_id)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Fetched {len(html):,} bytes of HTML")
        pairs = extract_flashcards_from_html(html)
        default_output = "flashcards.txt"
    else:
        # --- PDF mode ---
        if not Path(source).exists():
            print(f"Error: File not found: {source}", file=sys.stderr)
            sys.exit(1)
        print(f"Extracting flashcards from: {source}")
        pairs = extract_flashcards(source)
        default_output = str(Path(source).with_suffix('.txt'))

    if not pairs:
        print("No flashcard pairs found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} flashcard pairs")

    # Count types
    words = sum(1 for pt, en in pairs if ' ' not in pt and ' ' not in en)
    phrases = len(pairs) - words
    print(f"  - Single words: {words}")
    print(f"  - Phrases/sentences: {phrases}")

    if args.preview:
        print("\n--- PREVIEW ---")
        for i, (pt, en) in enumerate(pairs, 1):
            print(f"  {i:3d}. {pt}  ⟶  {en}")
        print("--- END PREVIEW ---\n")

    # Format and write output
    output = format_for_quizlet(pairs, args.delimiter)
    out_path = args.output if args.output else default_output

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\nSaved to: {out_path}")
    print(f"\nTo import into Quizlet:")
    print(f"  1. Go to quizlet.com → Create → + Create a new study set")
    print(f"  2. Click '+ Import' (top area)")
    print(f"  3. Paste the contents of {out_path}")
    print(f'  4. Set "Between term and definition" to Tab')
    print(f'  5. Set "Between rows" to New line')
    print(f"  6. Click Import")


if __name__ == "__main__":
    main()
