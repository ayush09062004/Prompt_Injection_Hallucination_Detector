"""
latex_parser/parser.py
Extracts structured elements from LaTeX source:
- Sections and their content
- Comments (% ...) 
- Macros (\\newcommand, \\def)
- Figure/table captions
- Citations and references
- BibTeX entries
"""

import re
from dataclasses import dataclass, field


@dataclass
class Section:
    """Represents a LaTeX section with its content."""
    level: str          # section, subsection, subsubsection
    title: str
    content: str
    start_line: int
    file: str = ""


@dataclass
class Comment:
    """A LaTeX comment line (% ...)."""
    text: str
    line_number: int
    file: str = ""


@dataclass
class Macro:
    """A LaTeX macro definition."""
    name: str
    definition: str
    raw: str
    line_number: int
    file: str = ""


@dataclass
class Caption:
    """A figure or table caption."""
    text: str
    env_type: str   # figure, table, algorithm, etc.
    line_number: int
    file: str = ""


@dataclass
class Citation:
    """A citation reference in the text."""
    key: str
    context: str    # surrounding text snippet
    line_number: int
    file: str = ""


@dataclass
class ParsedStructure:
    """Complete parsed structure of a LaTeX document."""
    sections: list[Section] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    macros: list[Macro] = field(default_factory=list)
    captions: list[Caption] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    bib_entries: dict[str, dict] = field(default_factory=dict)  # key → fields
    raw_text_by_section: dict[str, str] = field(default_factory=dict)
    plain_text: str = ""  # LaTeX stripped to plain text


class LaTeXParser:
    """
    Parses LaTeX source into structured components.
    Extracts sections, comments, macros, captions, and citations.
    """

    # Section hierarchy
    SECTION_LEVELS = ['chapter', 'section', 'subsection', 'subsubsection', 'paragraph']
    SECTION_RE = re.compile(
        r'\\(chapter|section|subsection|subsubsection|paragraph)\*?\{([^}]+)\}',
        re.MULTILINE
    )
    COMMENT_RE = re.compile(r'(?<!\\)%(.+?)$', re.MULTILINE)
    MACRO_NEWCMD_RE = re.compile(
        r'\\newcommand\{(\\[^}]+)\}(?:\[\d+\])?\{([^}]*(?:\{[^}]*\}[^}]*)*)\}',
        re.MULTILINE
    )
    MACRO_DEF_RE = re.compile(
        r'\\def\s*(\\[A-Za-z]+)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}',
        re.MULTILINE
    )
    CAPTION_RE = re.compile(r'\\caption\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', re.MULTILINE)
    CITE_RE = re.compile(r'\\cite[tp]?\{([^}]+)\}')
    ENV_RE = re.compile(r'\\begin\{(figure|table|algorithm|lstlisting)\}(.*?)\\end\{\1\}', re.DOTALL)

    def parse(self, tex_files: dict[str, str], bib_files: dict[str, str] = None) -> ParsedStructure:
        """Parse all .tex files and optional .bib files into ParsedStructure."""
        ps = ParsedStructure()

        for fname, content in tex_files.items():
            self._parse_comments(content, fname, ps)
            self._parse_macros(content, fname, ps)
            self._parse_captions(content, fname, ps)
            self._parse_citations(content, fname, ps)

        # Parse sections from the full resolved text or first tex file
        combined = "\n".join(tex_files.values())
        self._parse_sections(combined, ps)

        # Parse BibTeX
        if bib_files:
            for bname, bcontent in bib_files.items():
                self._parse_bibtex(bcontent, ps)

        # Build plain text representation
        ps.plain_text = self._strip_latex(combined)

        return ps

    def parse_resolved(self, resolved_text: str, bib_files: dict[str, str] = None) -> ParsedStructure:
        """Parse from a single resolved (\\input-expanded) text."""
        ps = ParsedStructure()
        self._parse_comments(resolved_text, "resolved", ps)
        self._parse_macros(resolved_text, "resolved", ps)
        self._parse_captions(resolved_text, "resolved", ps)
        self._parse_citations(resolved_text, "resolved", ps)
        self._parse_sections(resolved_text, ps)
        if bib_files:
            for bname, bcontent in bib_files.items():
                self._parse_bibtex(bcontent, ps)
        ps.plain_text = self._strip_latex(resolved_text)
        return ps

    def _parse_comments(self, content: str, fname: str, ps: ParsedStructure) -> None:
        """Extract all LaTeX comments (% lines)."""
        for i, line in enumerate(content.splitlines(), 1):
            # Find % not preceded by backslash
            m = re.search(r'(?<!\\)%(.*)$', line)
            if m:
                comment_text = m.group(1).strip()
                if comment_text:  # Non-empty comments only
                    ps.comments.append(Comment(
                        text=comment_text,
                        line_number=i,
                        file=fname
                    ))

    def _parse_macros(self, content: str, fname: str, ps: ParsedStructure) -> None:
        """Extract \\newcommand and \\def macro definitions."""
        lines = content.splitlines()
        line_starts = [0]
        pos = 0
        for line in lines:
            pos += len(line) + 1
            line_starts.append(pos)

        def get_line(match_start):
            for i, ls in enumerate(line_starts):
                if ls > match_start:
                    return i
            return len(lines)

        for m in self.MACRO_NEWCMD_RE.finditer(content):
            ps.macros.append(Macro(
                name=m.group(1),
                definition=m.group(2),
                raw=m.group(0),
                line_number=get_line(m.start()),
                file=fname
            ))

        for m in self.MACRO_DEF_RE.finditer(content):
            ps.macros.append(Macro(
                name=m.group(1),
                definition=m.group(2),
                raw=m.group(0),
                line_number=get_line(m.start()),
                file=fname
            ))

    def _parse_captions(self, content: str, fname: str, ps: ParsedStructure) -> None:
        """Extract captions with their environment context (figure/table)."""
        lines = content.splitlines()

        # Find captions within environments
        for env_match in self.ENV_RE.finditer(content):
            env_type = env_match.group(1)
            env_content = env_match.group(2)
            for cap_m in self.CAPTION_RE.finditer(env_content):
                # Compute line number (approximate)
                line_no = content[:env_match.start()].count('\n') + \
                          env_content[:cap_m.start()].count('\n') + 1
                ps.captions.append(Caption(
                    text=cap_m.group(1),
                    env_type=env_type,
                    line_number=line_no,
                    file=fname
                ))

        # Also catch captions outside environments
        for cap_m in self.CAPTION_RE.finditer(content):
            line_no = content[:cap_m.start()].count('\n') + 1
            # Avoid duplicates from environment pass
            if not any(c.line_number == line_no and c.file == fname for c in ps.captions):
                ps.captions.append(Caption(
                    text=cap_m.group(1),
                    env_type="unknown",
                    line_number=line_no,
                    file=fname
                ))

    def _parse_citations(self, content: str, fname: str, ps: ParsedStructure) -> None:
        """Extract \\cite{...} references with surrounding context."""
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            for m in self.CITE_RE.finditer(line):
                keys = [k.strip() for k in m.group(1).split(',')]
                for key in keys:
                    # Get context: the surrounding 100 chars
                    start = max(0, m.start() - 50)
                    end = min(len(line), m.end() + 50)
                    context = line[start:end].strip()
                    ps.citations.append(Citation(
                        key=key,
                        context=context,
                        line_number=i,
                        file=fname
                    ))

    def _parse_sections(self, content: str, ps: ParsedStructure) -> None:
        """Split document into sections and extract their text content."""
        # Find all section markers with positions
        markers = []
        for m in self.SECTION_RE.finditer(content):
            markers.append((m.start(), m.group(1), m.group(2)))

        for i, (start, level, title) in enumerate(markers):
            # Content goes from end of this marker to start of next
            end = markers[i + 1][0] if i + 1 < len(markers) else len(content)
            section_text = content[start:end]
            line_no = content[:start].count('\n') + 1

            section = Section(
                level=level,
                title=title.strip(),
                content=section_text,
                start_line=line_no,
            )
            ps.sections.append(section)
            ps.raw_text_by_section[f"{level}:{title.strip()}"] = section_text

    def _parse_bibtex(self, content: str, ps: ParsedStructure) -> None:
        """Parse BibTeX entries into structured dict."""
        # Match @type{key, ...}
        entry_re = re.compile(
            r'@(\w+)\s*\{\s*([^,]+),\s*(.*?)\n\}',
            re.DOTALL
        )
        field_re = re.compile(r'(\w+)\s*=\s*\{([^}]*)\}', re.DOTALL)

        for m in entry_re.finditer(content):
            entry_type = m.group(1).lower()
            entry_key = m.group(2).strip()
            fields_str = m.group(3)

            fields = {"type": entry_type}
            for fm in field_re.finditer(fields_str):
                fields[fm.group(1).lower()] = fm.group(2).strip()

            ps.bib_entries[entry_key] = fields

    def _strip_latex(self, content: str) -> str:
        """
        Remove LaTeX commands to get approximate plain text.
        Used for LLM analysis (reduces token count and noise).
        """
        # Remove comments
        text = re.sub(r'(?<!\\)%.*$', '', content, flags=re.MULTILINE)
        # Remove common environments
        text = re.sub(r'\\begin\{[^}]+\}|\\end\{[^}]+\}', '', text)
        # Remove common commands with arguments
        text = re.sub(r'\\(?:label|ref|cite[tp]?|href|url|footnote)\{[^}]*\}', '', text)
        # Remove formatting commands but keep content
        text = re.sub(r'\\(?:textbf|textit|emph|text|mathrm|mathbf)\{([^}]*)\}', r'\1', text)
        # Remove section commands but keep titles
        text = re.sub(r'\\(?:section|subsection|subsubsection|chapter|paragraph)\*?\{([^}]*)\}', r'\n\n### \1\n\n', text)
        # Remove remaining commands
        text = re.sub(r'\\[A-Za-z]+\*?(\{[^}]*\})*', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def get_section_chunks(self, ps: ParsedStructure, max_chars: int = 3000) -> list[dict]:
        """
        Break document into LLM-processable chunks by section.
        Returns list of {section_title, content, chunk_index}.
        """
        chunks = []
        for section in ps.sections:
            plain = self._strip_latex(section.content)
            # Split large sections into smaller chunks
            if len(plain) > max_chars:
                for i in range(0, len(plain), max_chars):
                    chunks.append({
                        "section_title": section.title,
                        "section_level": section.level,
                        "content": plain[i:i + max_chars],
                        "chunk_index": i // max_chars,
                        "file": section.file,
                    })
            else:
                chunks.append({
                    "section_title": section.title,
                    "section_level": section.level,
                    "content": plain,
                    "chunk_index": 0,
                    "file": section.file,
                })
        return chunks
