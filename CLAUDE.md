# CLAUDE.md

## Project
Portuguese Flashcard Extractor — pulls vocabulary, phrases and key expressions out of Portuguese lessons (PDFs or public Google Docs) and outputs Quizlet's import format.

## Layout
- `extract_flashcards.py` — CLI extractor
- `index.html` — browser front end

## Commands
- `python3 extract_flashcards.py lesson.pdf`
- `python3 extract_flashcards.py "<google-doc-url>" --output my_flashcards.txt`

## Notes
- Output must stay paste-compatible with Quizlet's Import feature.
