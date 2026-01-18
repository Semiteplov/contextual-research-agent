import gzip
import io
import re
import tarfile
from dataclasses import dataclass
from typing import ClassVar

from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedSource:
    arxiv_id: str
    main_tex_content: str
    all_tex_contents: dict[str, str]
    bib_content: str | None
    structure: list[dict]


class SourceExtractor:
    MAIN_FILE_PATTERNS: ClassVar[list[str]] = [
        r"main\.tex$",
        r"paper\.tex$",
        r"manuscript\.tex$",
        r"arxiv\.tex$",
        r".*\.tex$",
    ]

    def extract(self, content: bytes, arxiv_id: str) -> ExtractedSource:
        try:
            decompressed = gzip.decompress(content)
        except Exception as e:
            raise ValueError(f"Failed to decompress: {e}") from e

        return (
            self._extract_from_tar(decompressed, arxiv_id)
            if self._is_tar(decompressed)
            else self._extract_single_file(decompressed, arxiv_id)
        )

    def _is_tar(self, content: bytes) -> bool:
        return len(content) > 262 and content[257:262] == b"ustar"  # noqa: PLR2004

    def _extract_from_tar(self, content: bytes, arxiv_id: str) -> ExtractedSource:
        tex_files: dict[str, str] = {}
        bib_content: str | None = None

        with tarfile.open(fileobj=io.BytesIO(content), mode="r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                name = member.name.lower()

                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    file_content = f.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue

                if name.endswith(".tex"):
                    tex_files[member.name] = file_content
                elif name.endswith(".bib"):
                    bib_content = file_content

        if not tex_files:
            raise ValueError(f"No .tex files found in archive for {arxiv_id}")

        main_file = self._find_main_file(tex_files)
        main_content = tex_files[main_file]

        full_content = self._resolve_includes(main_content, tex_files)

        structure = self._parse_structure(full_content)

        return ExtractedSource(
            arxiv_id=arxiv_id,
            main_tex_content=full_content,
            all_tex_contents=tex_files,
            bib_content=bib_content,
            structure=structure,
        )

    def _extract_single_file(self, content: bytes, arxiv_id: str) -> ExtractedSource:
        tex_content = content.decode("utf-8", errors="ignore")
        structure = self._parse_structure(tex_content)

        return ExtractedSource(
            arxiv_id=arxiv_id,
            main_tex_content=tex_content,
            all_tex_contents={"main.tex": tex_content},
            bib_content=None,
            structure=structure,
        )

    def _find_main_file(self, tex_files: dict[str, str]) -> str:
        for pattern in self.MAIN_FILE_PATTERNS:
            for filename, content in tex_files.items():
                if re.search(pattern, filename.lower()) and r"\documentclass" in content:
                    return filename

        for filename, content in tex_files.items():
            if r"\documentclass" in content:
                return filename

        return next(iter(tex_files))

    def _resolve_includes(
        self,
        content: str,
        tex_files: dict[str, str],
    ) -> str:
        pattern = r"\\(?:input|include)\{([^}]+)\}"

        def replace_include(match: re.Match) -> str:
            filename = match.group(1)
            if not filename.endswith(".tex"):
                filename += ".tex"

            for tex_name, tex_content in tex_files.items():
                if tex_name.endswith(filename) or tex_name == filename:
                    return tex_content

            return f"% [MISSING: {filename}]"

        return re.sub(pattern, replace_include, content)

    def _parse_structure(self, content: str) -> list[dict]:
        structure = []

        content = re.sub(r"%.*$", "", content, flags=re.MULTILINE)
        section_pattern = r"\\(section|subsection|subsubsection)\*?\{([^}]+)\}"

        for match in re.finditer(section_pattern, content):
            level = match.group(1)
            title = match.group(2)
            title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
            title = re.sub(r"\\[a-zA-Z]+", "", title)
            title = title.strip()

            structure.append(
                {
                    "level": level,
                    "title": title,
                    "position": match.start(),
                }
            )

        return structure


class LaTeXToTextConverter:
    REMOVE_COMMANDS: ClassVar[list[str]] = [
        r"\\begin\{figure\}.*?\\end\{figure\}",
        r"\\begin\{table\}.*?\\end\{table\}",
        r"\\begin\{algorithm\}.*?\\end\{algorithm\}",
        r"\\bibliographystyle\{[^}]*\}",
        r"\\bibliography\{[^}]*\}",
        r"\\usepackage\{[^}]*\}",
        r"\\documentclass.*?\n",
        r"\\author\{[^}]*\}",
        r"\\affiliation\{[^}]*\}",
        r"\\email\{[^}]*\}",
        r"\\date\{[^}]*\}",
    ]

    UNWRAP_COMMANDS: ClassVar[list[tuple[str, str]]] = [
        (r"\\textbf\{([^}]*)\}", r"\1"),
        (r"\\textit\{([^}]*)\}", r"\1"),
        (r"\\emph\{([^}]*)\}", r"\1"),
        (r"\\text\{([^}]*)\}", r"\1"),
        (r"\\href\{[^}]*\}\{([^}]*)\}", r"\1"),
        (r"\\footnote\{([^}]*)\}", r" [\1]"),
        (r"\\cite\{([^}]*)\}", r"[CITE:\1]"),
        (r"\\ref\{([^}]*)\}", r"[REF:\1]"),
        (r"\\label\{[^}]*\}", ""),
    ]

    def convert(self, latex: str) -> str:
        text = latex

        text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)

        doc_match = re.search(r"\\begin\{document\}", text)
        if doc_match:
            text = text[doc_match.end() :]

        text = re.sub(r"\\end\{document\}.*", "", text, flags=re.DOTALL)

        for pattern in self.REMOVE_COMMANDS:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r"\$\$(.+?)\$\$", r"[MATH_DISPLAY:\1]", text, flags=re.DOTALL)
        text = re.sub(r"\$(.+?)\$", r"[MATH_INLINE:\1]", text)
        text = re.sub(
            r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}",
            r"[MATH_EQUATION:\1]",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"\\begin\{align\*?\}(.+?)\\end\{align\*?\}",
            r"[MATH_ALIGN:\1]",
            text,
            flags=re.DOTALL,
        )

        for pattern, replacement in self.UNWRAP_COMMANDS:
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"\\section\*?\{([^}]+)\}", r"\n\n# \1\n\n", text)
        text = re.sub(r"\\subsection\*?\{([^}]+)\}", r"\n\n## \1\n\n", text)
        text = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"\n\n### \1\n\n", text)
        text = re.sub(r"\\begin\{itemize\}", "\n", text)
        text = re.sub(r"\\end\{itemize\}", "\n", text)
        text = re.sub(r"\\begin\{enumerate\}", "\n", text)
        text = re.sub(r"\\end\{enumerate\}", "\n", text)
        text = re.sub(r"\\item\s*", "\n• ", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{[^}]*\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
        text = re.sub(r"\{|\}", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text)

        return text.strip()
