"""
ingestion/ingestor.py
Handles ZIP extraction and recursive LaTeX file resolution.
Resolves \\input and \\include directives to build a unified document view.
"""

import zipfile
import os
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """Container for all parsed content from a LaTeX ZIP."""
    # Maps filename → raw content
    tex_files: dict[str, str] = field(default_factory=dict)
    bib_files: dict[str, str] = field(default_factory=dict)
    other_files: list[str] = field(default_factory=list)
    # Unified resolved text (after \\input/\\include expansion)
    resolved_text: str = ""
    # Root .tex file name
    root_file: str = ""
    # All included file paths in order
    include_chain: list[str] = field(default_factory=list)
    # Extraction temp dir (caller responsible for cleanup)
    extract_dir: str = ""


class LaTeXIngestor:
    """
    Extracts a ZIP archive and produces a ParsedDocument with:
    - All .tex files parsed individually (for source attribution)
    - A unified resolved text where \\input/\\include are expanded inline
    - .bib file contents for citation verification
    """

    # Patterns for file inclusion directives
    INPUT_RE = re.compile(r'\\(?:input|include)\{([^}]+)\}')

    def __init__(self, max_file_size_mb: int = 50):
        self.max_bytes = max_file_size_mb * 1024 * 1024

    def ingest(self, zip_path: str) -> ParsedDocument:
        """
        Main entry point. Extract ZIP, find root .tex, resolve all includes.
        Returns a fully populated ParsedDocument.
        """
        doc = ParsedDocument()

        # Create temp dir for extraction
        tmp_dir = tempfile.mkdtemp(prefix="latex_detector_")
        doc.extract_dir = tmp_dir

        # Extract ZIP safely (path traversal protection)
        self._safe_extract(zip_path, tmp_dir)

        # Collect all .tex and .bib files
        for root, _, files in os.walk(tmp_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, tmp_dir)
                ext = fname.lower().split('.')[-1] if '.' in fname else ''

                if ext == 'tex':
                    content = self._read_file(fpath)
                    doc.tex_files[rel] = content
                elif ext == 'bib':
                    content = self._read_file(fpath)
                    doc.bib_files[rel] = content
                else:
                    doc.other_files.append(rel)

        # Identify root .tex file (one with \\documentclass)
        doc.root_file = self._find_root(doc.tex_files)

        # Recursively resolve \\input / \\include into unified text
        visited = set()
        doc.resolved_text = self._resolve_includes(
            doc.root_file, doc.tex_files, tmp_dir, visited, doc.include_chain
        )

        return doc

    def _safe_extract(self, zip_path: str, dest_dir: str) -> None:
        """Extract ZIP with path traversal protection."""
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.infolist():
                # Sanitize path — prevent directory traversal attacks
                member_path = os.path.normpath(member.filename)
                if member_path.startswith('..') or member_path.startswith('/'):
                    continue  # Skip dangerous paths
                dest_path = os.path.join(dest_dir, member_path)
                # Prevent extraction outside dest_dir
                if not dest_path.startswith(dest_dir):
                    continue
                zf.extract(member, dest_dir)

    def _read_file(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file with size limit and encoding fallback."""
        try:
            size = os.path.getsize(path)
            if size > self.max_bytes:
                return f"[FILE TOO LARGE: {size} bytes, skipped]"
            with open(path, 'r', encoding=encoding, errors='replace') as f:
                return f.read()
        except Exception as e:
            return f"[READ ERROR: {e}]"

    def _find_root(self, tex_files: dict[str, str]) -> str:
        """
        Find the root .tex file by looking for \\documentclass.
        Falls back to the alphabetically first .tex if not found.
        """
        for fname, content in tex_files.items():
            if '\\documentclass' in content:
                return fname
        # Fallback: return first available
        return next(iter(tex_files), "")

    def _resolve_includes(
        self,
        filename: str,
        tex_files: dict[str, str],
        base_dir: str,
        visited: set,
        chain: list[str],
        depth: int = 0,
    ) -> str:
        """
        Recursively expand \\input{file} and \\include{file} directives.
        Prevents infinite loops via visited set. Max depth = 10.
        """
        if depth > 10:
            return f"% [MAX INCLUDE DEPTH REACHED for {filename}]\n"
        if filename in visited:
            return f"% [CIRCULAR INCLUDE SKIPPED: {filename}]\n"

        visited.add(filename)

        # Find the file content - try multiple path resolutions
        content = tex_files.get(filename)
        if content is None:
            # Try without extension
            for key in tex_files:
                if key.endswith(f"/{filename}") or key == filename:
                    content = tex_files[key]
                    filename = key
                    break
        if content is None:
            return f"% [MISSING FILE: {filename}]\n"

        chain.append(filename)

        # Replace \\input{...} and \\include{...} with resolved content
        def replacer(match):
            inc_name = match.group(1).strip()
            # Add .tex extension if missing
            if not inc_name.endswith('.tex'):
                inc_name_tex = inc_name + '.tex'
            else:
                inc_name_tex = inc_name

            # Try to find in tex_files by various path patterns
            resolved = None
            for try_name in [inc_name_tex, inc_name, os.path.basename(inc_name_tex)]:
                if try_name in tex_files:
                    resolved = try_name
                    break
                # Try subdirectory match
                for key in tex_files:
                    if key.endswith('/' + try_name) or key.endswith('\\' + try_name):
                        resolved = key
                        break
                if resolved:
                    break

            if resolved:
                return (
                    f"\n% ===== BEGIN INCLUDE: {resolved} =====\n"
                    + self._resolve_includes(resolved, tex_files, base_dir, visited, chain, depth + 1)
                    + f"\n% ===== END INCLUDE: {resolved} =====\n"
                )
            else:
                return f"\n% [UNRESOLVED INCLUDE: {inc_name}]\n"

        return self.INPUT_RE.sub(replacer, content)

    def get_file_line_map(self, tex_files: dict[str, str]) -> dict[str, list[str]]:
        """Return line-by-line map for each tex file (for line number reporting)."""
        return {fname: content.splitlines() for fname, content in tex_files.items()}
