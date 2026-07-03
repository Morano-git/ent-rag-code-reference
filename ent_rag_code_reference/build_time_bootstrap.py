"""
Bootstrap the ENT RAG build-time implementation.

Run from the project root:
    python bootstrap_build_time_v2.py

This script creates:
    build_time/scripts/      Python implementation files
    build_time/rag_digest/   Generated artifact folders

Unlike the earlier scaffold-only script, this version writes executable module code
adapted from the working rapid-development notebook.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_TIME_DIR = PROJECT_ROOT / "build_time"
RAG_DIGEST_SCRIPT_DIR = BUILD_TIME_DIR / "scripts"
RAG_DIGEST_ARTIFACTS_DIR = BUILD_TIME_DIR / "rag_digest"

# Do you want to reset the directory every run?
RESET_BUILD_TIME_DIR = True
# Do you want to overwrite the existing modules every run?
OVERWRITE_MODULES = True


# RAG Digest Main File
RAG_DIGEST_SCRIPT_EXEC_PY_PATH = RAG_DIGEST_SCRIPT_DIR / "rag_digest.py"

DIRECTORIES = [
    BUILD_TIME_DIR,
    RAG_DIGEST_SCRIPT_DIR,
    RAG_DIGEST_ARTIFACTS_DIR,
    RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities",
    RAG_DIGEST_SCRIPT_DIR / "semantic_chunking_utilities",
    RAG_DIGEST_SCRIPT_DIR / "indexing_utilities",
    RAG_DIGEST_ARTIFACTS_DIR / "raw_images",
    RAG_DIGEST_ARTIFACTS_DIR / "lookup_tables",
    RAG_DIGEST_ARTIFACTS_DIR / "metadata",
    RAG_DIGEST_ARTIFACTS_DIR / "faiss_indexes",
    RAG_DIGEST_ARTIFACTS_DIR / "embedding_matrices",
    RAG_DIGEST_ARTIFACTS_DIR / "reports",
]

FILES = {}

FILES[RAG_DIGEST_SCRIPT_DIR / "__init__.py"] = ""
FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "__init__.py"] = ""
FILES[RAG_DIGEST_SCRIPT_DIR / "semantic_chunking_utilities" / "__init__.py"] = ""
FILES[RAG_DIGEST_SCRIPT_DIR / "indexing_utilities" / "__init__.py"] = ""

FILES[RAG_DIGEST_SCRIPT_DIR / "config.py"] = dedent(r'''
    """Configuration values for the build-time RAG digest pipeline."""

    from pathlib import Path


    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # 1. PDF Source
    PDF_PATH = PROJECT_ROOT / "pdfs" / "SFO_UK_Handbook_for_ENT_Reformat.pdf"
    
    # 2. Build Time Path
    BUILD_TIME_DIR = PROJECT_ROOT / "build_time"
    
    # 2.1. Artifact Script
    SCRIPT_DIR = BUILD_TIME_DIR / "scripts"
    
    # 2.2. Artifact Directory
    ARTIFACTS_DIR = BUILD_TIME_DIR / "rag_digest"
    
    RAW_IMAGES_DIR = ARTIFACTS_DIR / "raw_images"
    LOOKUP_TABLES_DIR = ARTIFACTS_DIR / "lookup_tables"
    METADATA_DIR = ARTIFACTS_DIR / "metadata"
    FAISS_INDEXES_DIR = ARTIFACTS_DIR / "faiss_indexes"
    EMBEDDING_MATRICES_DIR = ARTIFACTS_DIR / "embedding_matrices"
    REPORTS_DIR = ARTIFACTS_DIR / "reports"

    # Encoder Model Names
    BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    
    # Semantic Chunker Parameters
    SEMANTIC_BUFFER_SIZE = 1
    SEMANTIC_BREAKPOINT_PERCENTILE_THRESHOLD = 95
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "data_models.py"] = dedent(r'''
    """Dataclasses used by the subsection chunking stage."""

    from __future__ import annotations

    from dataclasses import dataclass
    from typing import Optional, Tuple


    @dataclass
    class HeaderRecord:
        header_id: int
        hierarchy: int
        title: str
        page_index: int
        page_number: int


    @dataclass
    class TableRecord:
        table_id: int
        page_index: int
        bbox: Tuple[float, float, float, float]
        text: str


    @dataclass
    class ImageRecord:
        image_id: int
        page_index: int
        bbox: Tuple[float, float, float, float]
        image_path: Optional[str]
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "text_reconstruction.py"] = dedent(r'''
    """Text-block reconstruction and subsection passage normalization utilities."""

    from __future__ import annotations

    import re
    from typing import List


    def normalize_space(text: str) -> str:
        """Normalize whitespace while preserving paragraph boundaries."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u200b", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


    BULLET_PATTERN = re.compile(r"^[●•▪▫◦]\s*$")
    BULLET_ITEM_PATTERN = re.compile(r"^[●•▪▫◦]\s+")


    def clean_text_line(line: str) -> str:
        line = line.replace("\u200b", "")
        line = re.sub(r"[ \t]+", " ", line)
        return line.strip()


    def is_empty_line(line: str) -> bool:
        return clean_text_line(line) == ""


    def is_bullet_line(line: str) -> bool:
        return bool(BULLET_PATTERN.fullmatch(clean_text_line(line)))


    def is_bullet_item_line(line: str) -> bool:
        return bool(BULLET_ITEM_PATTERN.match(clean_text_line(line)))


    def construct_text_block(raw_text: str) -> str:
        """
        Repair PyMuPDF bullet extraction within one text block.

        Common pattern:
            ●
            item line 1
            item line 2

        Becomes:
            ● item line 1 item line 2
        """
        raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        raw_text = raw_text.replace("\u200b", "")
        lines = raw_text.split("\n")

        if not any(clean_text_line(line) for line in lines):
            return "\n"

        constructed_lines: List[str] = []
        i = 0

        while i < len(lines):
            line = clean_text_line(lines[i])

            if line == "":
                constructed_lines.append("")
                i += 1
                continue

            if is_bullet_line(line):
                bullet_symbol = line

                if i + 1 >= len(lines) or is_empty_line(lines[i + 1]):
                    constructed_lines.append(bullet_symbol)
                    i += 1
                    continue

                item_lines = []
                j = i + 1

                while j < len(lines):
                    next_line = clean_text_line(lines[j])

                    if next_line == "":
                        break

                    if is_bullet_line(next_line):
                        break

                    item_lines.append(next_line)
                    j += 1

                bullet_text = " ".join(item_lines).strip()
                constructed_lines.append(f"{bullet_symbol} {bullet_text}" if bullet_text else bullet_symbol)
                i = j
                continue

            constructed_lines.append(line)
            i += 1

        text = "\n".join(constructed_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip("\n")


    def is_img_marker_line(line: str) -> bool:
        return bool(re.fullmatch(r"<img\s+\d+>", clean_text_line(line)))


    def is_table_block(paragraph: str) -> bool:
        paragraph = paragraph.strip()
        
        return bool(re.fullmatch(r"<t\s+\d+>\n.*?\n</t>", paragraph, flags=re.DOTALL))


    def protect_hard_markers(text: str) -> str:
        """Force image/table markers to become hard paragraph boundaries."""
        text = re.sub(r"\n*\s*(<img\s+\d+>)\s*\n*", r"\n\n\1\n\n", text)
        text = re.sub(r"\n*(<t\s+\d+>\n.*?\n</t>)\n*", r"\n\n\1\n\n", text, flags=re.DOTALL)
        
        return text


    def merge_single_newline_paragraph(paragraph: str) -> str:
        """Merge line-wrapped captions/paragraphs while preserving bullet items."""
        paragraph = paragraph.strip()

        if not paragraph:
            return ""

        if is_img_marker_line(paragraph):
            return paragraph

        if is_table_block(paragraph):
            return paragraph

        lines = [clean_text_line(line) for line in paragraph.split("\n") if clean_text_line(line) != ""]

        if not lines:
            return ""

        merged_lines = []
        buffer = []

        def flush_buffer() -> None:
            nonlocal buffer
            if buffer:
                merged_lines.append(" ".join(buffer).strip())
                buffer = []

        for idx, line in enumerate(lines):
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""

            if is_bullet_item_line(line):
                flush_buffer()
                merged_lines.append(line)
                continue

            if next_line and is_bullet_item_line(next_line):
                flush_buffer()
                merged_lines.append(line)
                continue

            if is_img_marker_line(line):
                flush_buffer()
                merged_lines.append(line)
                continue

            buffer.append(line)

        flush_buffer()
        
        return "\n".join(line for line in merged_lines if line.strip())


    def normalize_subsection_passage(text: str) -> str:
        """Final subsection-level cleanup after all page blocks have been appended."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u200b", "")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = protect_hard_markers(text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        paragraphs = text.split("\n\n")
        normalized_paragraphs = []

        for paragraph in paragraphs:
            normalized = merge_single_newline_paragraph(paragraph)
            if normalized.strip():
                normalized_paragraphs.append(normalized)

        text = "\n\n".join(normalized_paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "pdf_outline.py"] = dedent(r'''
    """PDF outline and header extraction utilities."""

    from __future__ import annotations

    from typing import List

    import fitz

    from subsection_chunking_utilities.data_models import HeaderRecord
    from subsection_chunking_utilities.text_reconstruction import normalize_space


    def flatten_headers(doc: fitz.Document) -> List[HeaderRecord]:
        """Flatten all PDF outline entries into sequential subsection-level headers."""
        raw_toc = doc.get_toc(simple=True)
        headers: List[HeaderRecord] = []

        for idx, item in enumerate(raw_toc):
            level, title, page_number = item
            headers.append(
                HeaderRecord(
                    header_id=idx,
                    hierarchy=int(level),
                    title=normalize_space(str(title)),
                    page_index=int(page_number) - 1,
                    page_number=int(page_number),
                )
            )

        return headers
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "block_detection.py"] = dedent(r'''
    """PyMuPDF block detection helpers."""

    from __future__ import annotations

    from typing import Any, Dict, List, Tuple

    import fitz

    from subsection_chunking_utilities.text_reconstruction import construct_text_block


    def find_bbox_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


    def text_block_centroid_inside_tbl_bbox(
            centroid: Tuple[float, float],
            ref_tbl: Tuple[float, float, float, float],
            margin: float = 1.0,
        ) -> bool:
        x, y = centroid
        x0, y0, x1, y1 = ref_tbl
        
        return (x0 - margin) <= x <= (x1 + margin) and (y0 - margin) <= y <= (y1 + margin)


    def extract_block_text(block: Dict[str, Any]) -> str:
        """Extract visible text from a PyMuPDF text block."""
        if block.get("type") != 0:
            return ""

        raw_lines = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            raw_lines.append(line_text)

        raw_text = "\n".join(raw_lines)
        
        return construct_text_block(raw_text)


    def preprocess_block_detection(page: fitz.Page) -> List[Dict[str, Any]]:
        """Return text/image blocks in a deterministic top-down, left-right order."""
        try:
            data = page.get_text("dict", sort=True)
        except TypeError:
            data = page.get_text("dict")

        blocks = sorted(
            data.get("blocks", []),
            key=lambda block: (
                round(block.get("bbox", [0, 0, 0, 0])[1], 1),
                round(block.get("bbox", [0, 0, 0, 0])[0], 1),
            ),
        )
        
        return blocks
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "table_processing.py"] = dedent(r'''
    """Table detection, normalization, and suppression utilities."""

    from __future__ import annotations
    
    # Intended to suppress: "Consider using the pymupdf_layout package for a greatly improved page layout analysis."
    import io
    from contextlib import redirect_stdout, redirect_stderr

    from typing import Any, List, Optional, Tuple

    import fitz

    from subsection_chunking_utilities.block_detection import (
        find_bbox_centroid,
        text_block_centroid_inside_tbl_bbox,
    )
    from subsection_chunking_utilities.data_models import TableRecord
    from subsection_chunking_utilities.text_reconstruction import normalize_space


    def normalize_table_cells(table_cells: List[List[Any]]) -> str:
        """Convert extracted table cells into a compact text form."""
        rows = []
        for row in table_cells:
            clean_cells = []
            for cell in row:
                clean_cells.append("" if cell is None else normalize_space(str(cell)))
            row_text = " | ".join(clean_cells).strip()
            if row_text:
                rows.append(row_text)
        
        return "\n".join(rows).strip()


    def detect_tables_on_page(
            page: fitz.Page,
            page_index: int,
            starting_table_id: int = 0,
        ) -> Tuple[List[TableRecord], int]:
        """Detect tables on a page and return normalized table records."""
        table_records: List[TableRecord] = []
        table_counter = starting_table_id

        if not hasattr(page, "find_tables"):
            return table_records, table_counter

        try:
            # Tries to extract table embeddings while suppressing the PyMuPDF warning
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                table_finder = page.find_tables()
            tables = getattr(table_finder, "tables", [])

        except Exception as exc:
            print(f"[WARN] Table detection failed on page {page_index + 1}: {exc}")
            return table_records, table_counter

        for table in tables:
            table_counter += 1

            try:
                cells = table.extract()
                table_text = normalize_table_cells(cells)
            except Exception:
                table_text = ""

            if not table_text:
                table_text = "[table detected but text extraction failed]"

            table_records.append(
                TableRecord(
                    table_id=table_counter,
                    page_index=page_index,
                    bbox=tuple(table.bbox),
                    text=table_text,
                )
            )

        return table_records, table_counter


    def find_containing_table(
            block_bbox: Tuple[float, float, float, float],
            table_records: List[TableRecord],
        ) -> Optional[TableRecord]:
        """Return the table whose bbox contains the block centroid, if any."""
        center = find_bbox_centroid(block_bbox)
        for table in table_records:
            if text_block_centroid_inside_tbl_bbox(center, table.bbox):
                return table
        
        return None
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "image_extraction.py"] = dedent(r'''
    """PDF image extraction utilities."""

    from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict

    import fitz

    from subsection_chunking_utilities.data_models import ImageRecord


    def save_image_block(
            page: fitz.Page,
            block: Dict[str, Any],
            image_id: int,
            doc_stem: str,
            image_dir: Path,
        ) -> ImageRecord:
        """Save one PyMuPDF image block and return its image record."""
        page_index = page.number
        bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        ext = block.get("ext", "png")
        ext = ext.lower().replace(".", "")

        image_filename = f"{doc_stem}_p{page_index + 1:03d}_img{image_id:04d}.{ext}"
        image_path = image_dir / image_filename
        image_bytes = block.get("image", None)

        if image_bytes:
            image_path.write_bytes(image_bytes)
        else:
            pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=200)
            image_path = image_path.with_suffix(".png")
            pix.save(image_path)

        return ImageRecord(
            image_id=image_id,
            page_index=page_index,
            bbox=bbox,
            image_path=str(image_path),
        )
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_utilities" / "page_extraction.py"] = dedent(r'''
    """Subsection-level page traversal and passage construction."""

    from __future__ import annotations

    import re
    from dataclasses import asdict
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import fitz

    from subsection_chunking_utilities.block_detection import extract_block_text, preprocess_block_detection
    from subsection_chunking_utilities.data_models import HeaderRecord
    from subsection_chunking_utilities.image_extraction import save_image_block
    from subsection_chunking_utilities.pdf_outline import flatten_headers
    from subsection_chunking_utilities.table_processing import detect_tables_on_page, find_containing_table
    from subsection_chunking_utilities.text_reconstruction import normalize_subsection_passage


    def make_subsection_key(header: HeaderRecord) -> str:
        return f"{header.header_id:03d}::{header.title}"


    def finalize_subsection_builder(
            subsection_chunk: Dict[str, Dict[str, Any]],
            current_header: Optional[HeaderRecord],
            page_blocks_string_builder: List[str],
        ) -> List[str]:
        if current_header is None:
            return []

        subsection_key = make_subsection_key(current_header)
        passage = "\n".join(part for part in page_blocks_string_builder if part is not None)
        passage = normalize_subsection_passage(passage)

        subsection_chunk[subsection_key] = {
            **asdict(current_header),
            "subsection_key": subsection_key,
            "char_count": len(passage),
            "image_marker_count": len(re.findall(r"<img\s+\d+>", passage)),
            "table_marker_count": len(re.findall(r"<t\s+\d+>", passage)),
            "passage": passage,
        }

        return []


    def header_iteration_logic(
            page_index: int,
            headers: List[HeaderRecord],
            h_index: int,
            current_header: Optional[HeaderRecord],
            subsection_chunk: Dict[str, Dict[str, Any]],
            page_blocks_string_builder: List[str],
        ) -> Tuple[Optional[HeaderRecord], int, List[str]]:
        """
        Page-boundary header transition logic.

        Assumption:
            Each next flattened section/subsection begins on a new page.
        """
        while h_index < len(headers) and page_index >= headers[h_index].page_index:
            page_blocks_string_builder = finalize_subsection_builder(
                subsection_chunk=subsection_chunk,
                current_header=current_header,
                page_blocks_string_builder=page_blocks_string_builder,
            )
            current_header = headers[h_index]
            h_index += 1

        return current_header, h_index, page_blocks_string_builder


    def build_subsection_passages(
            doc_path: Path,
            image_dir: Path,
        ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """
        Build subsection-level passages and records for extracted images/tables.

        Returns:
            subsection_chunk: {subsection_key: subsection metadata and passage}
            image_records: {image_id: image metadata}
            table_records: {table_id: table metadata}
        """
        image_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(doc_path)
        doc_stem = doc_path.stem

        try:
            headers = flatten_headers(doc)
            if not headers:
                raise ValueError("No PDF outline / TOC entries found.")

            headers = sorted(headers, key=lambda h: (h.page_index, h.header_id))

            subsection_chunk: Dict[str, Dict[str, Any]] = {}
            image_records: Dict[int, Dict[str, Any]] = {}
            table_records: Dict[int, Dict[str, Any]] = {}

            figure_tracker = {"img": 0, "tbl": 0}

            h_index = 0
            current_header: Optional[HeaderRecord] = None
            page_blocks_string_builder: List[str] = []

            first_outline_page = headers[0].page_index
            for page_index in range(first_outline_page, len(doc)):
                page = doc[page_index]

                current_header, h_index, page_blocks_string_builder = header_iteration_logic(
                    page_index=page_index,
                    headers=headers,
                    h_index=h_index,
                    current_header=current_header,
                    subsection_chunk=subsection_chunk,
                    page_blocks_string_builder=page_blocks_string_builder,
                )

                if current_header is None:
                    continue

                page_tables, figure_tracker["tbl"] = detect_tables_on_page(
                    page=page,
                    page_index=page_index,
                    starting_table_id=figure_tracker["tbl"],
                )

                for table in page_tables:
                    table_records[table.table_id] = {
                        **asdict(table),
                        "subsection_key": make_subsection_key(current_header),
                    }

                emitted_table_ids_on_page = set()
                blocks = preprocess_block_detection(page)

                for block in blocks:
                    block_type = block.get("type")
                    block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))

                    if block_type == 0:
                        containing_table = find_containing_table(
                            block_bbox=block_bbox,
                            table_records=page_tables,
                        )

                        if containing_table is not None:
                            table_id = containing_table.table_id
                            if table_id not in emitted_table_ids_on_page:
                                table_markup = f"\n\n<t {table_id}>\n{containing_table.text}\n</t>\n\n"
                                page_blocks_string_builder.append(table_markup)
                                emitted_table_ids_on_page.add(table_id)
                            continue

                        text = extract_block_text(block)

                        if text.strip() == "":
                            page_blocks_string_builder.append("\n")
                            continue

                        page_blocks_string_builder.append(text)

                    elif block_type == 1:
                        figure_tracker["img"] += 1
                        image_id = figure_tracker["img"]

                        try:
                            img_record = save_image_block(
                                page=page,
                                block=block,
                                image_id=image_id,
                                doc_stem=doc_stem,
                                image_dir=image_dir,
                            )
                            image_records[image_id] = {
                                **asdict(img_record),
                                "subsection_key": make_subsection_key(current_header),
                                "error": None,
                            }
                        except Exception as exc:
                            print(f"[WARN] Failed to save img_{image_id} on page {page_index + 1}: {exc}")
                            image_records[image_id] = {
                                "image_id": image_id,
                                "page_index": page_index,
                                "bbox": block_bbox,
                                "image_path": None,
                                "subsection_key": make_subsection_key(current_header),
                                "error": str(exc),
                            }

                        page_blocks_string_builder.append(f"\n<img {image_id}>\n")

            page_blocks_string_builder = finalize_subsection_builder(
                subsection_chunk=subsection_chunk,
                current_header=current_header,
                page_blocks_string_builder=page_blocks_string_builder,
            )

        finally:
            doc.close()

        return subsection_chunk, image_records, table_records
''').strip() + "\n"

# Subsection-level chunking orchestrator
FILES[RAG_DIGEST_SCRIPT_DIR / "subsection_chunking_module.py"] = dedent(r'''
    """Subsection-level PDF chunking orchestrator."""

    from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict, Tuple

    from config import PDF_PATH, RAW_IMAGES_DIR
    from subsection_chunking_utilities.page_extraction import build_subsection_passages


    def run_subsection_chunking(
        pdf_path: Path = PDF_PATH,
        image_dir: Path = RAW_IMAGES_DIR,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """Run the first loop: PDF -> subsection passages + image/table records."""
        print("[1/4] Running subsection chunking")
        subsection_chunk, image_records, table_records = build_subsection_passages(
            doc_path=Path(pdf_path),
            image_dir=Path(image_dir),
        )

        print(f"\tSubsection records: {len(subsection_chunk)}")
        print(f"\tImage records:      {len(image_records)}")
        print(f"\tTable records:      {len(table_records)}")

        return subsection_chunk, image_records, table_records
''').strip() + "\n"


FILES[RAG_DIGEST_SCRIPT_DIR / "semantic_chunking_utilities" / "passage_preprocessing.py"] = dedent(r'''
    """Passage preprocessing utilities before semantic chunking."""

    from __future__ import annotations

    import re


    IMAGE_MARKER_RE = re.compile(r"<img\s+(\d+)>")


    def clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u200b", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()


    def prepare_text_for_semantic_chunking(s_passage: str) -> str:
        """
        Prepare subsection passage before semantic chunking.

        Policy:
            - remove <img N> markers
            - keep figure captions
            - preserve table text but remove <t N> wrappers
        """
        text = IMAGE_MARKER_RE.sub("", s_passage)
        text = re.sub(r"<t\s+\d+>\n", "", text).replace("\n</t>", "")
        text = re.sub(r"\n{3,}", "\n\n", text)

        return clean_text(text)
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "semantic_chunking_utilities" / "image_contexts.py"] = dedent(r'''
    """Image marker, caption, antecedent, and subsequent passage utilities."""

    from __future__ import annotations

    import re
    from typing import Any, Dict, List, Optional, Tuple

    from semantic_chunking_utilities.passage_preprocessing import IMAGE_MARKER_RE, clean_text


    FIGURE_CAPTION_RE = re.compile(
        r"^\s*(Figure\s+\d+[A-Za-z]?\s*:\s*.*?)(?=\n\n|<img\s+\d+>|$)",
        flags=re.DOTALL,
    )

    TABLE_BLOCK_RE = re.compile(r"<t\s+\d+>\n.*?\n</t>", flags=re.DOTALL)


    def paragraph_is_image_marker(paragraph: str) -> bool:
        return bool(re.fullmatch(r"<img\s+\d+>", paragraph.strip()))


    def paragraph_is_figure_caption(paragraph: str) -> bool:
        return bool(re.match(r"^Figure\s+\d+[A-Za-z]?\s*:", paragraph.strip()))


    def paragraph_is_table_block(paragraph: str) -> bool:
        return bool(TABLE_BLOCK_RE.fullmatch(paragraph.strip()))


    def extract_image_caption(after_img_text: str) -> Tuple[Optional[str], str]:
        """Extract immediate Figure caption after an <img N> marker."""
        after_img_text = after_img_text.lstrip()
        match = FIGURE_CAPTION_RE.match(after_img_text)
        if not match:
            return None, after_img_text

        caption = clean_text(match.group(1))
        remaining = after_img_text[match.end():].lstrip()
        return caption, remaining


    def get_last_antecedent_passage(before_img_text: str) -> Optional[str]:
        """Get nearest paragraph before image marker, skipping image markers/captions."""
        paragraphs = [clean_text(p) for p in before_img_text.split("\n\n") if clean_text(p)]
        for paragraph in reversed(paragraphs):
            if paragraph_is_image_marker(paragraph) or paragraph_is_figure_caption(paragraph):
                continue
            return paragraph
        return None


    def get_first_subsequent_passage(after_caption_text: str) -> Optional[str]:
        """Get nearest paragraph after image caption, skipping image markers/captions."""
        paragraphs = [clean_text(p) for p in after_caption_text.split("\n\n") if clean_text(p)]
        for paragraph in paragraphs:
            if paragraph_is_image_marker(paragraph) or paragraph_is_figure_caption(paragraph):
                continue
            return paragraph
        return None

    def extract_image_contexts_from_subsection(
            s_passage: str,
            image_records: Dict[Any, Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
        """Extract all <img N> contexts from one subsection passage."""
        image_contexts = []

        for match in IMAGE_MARKER_RE.finditer(s_passage):
            fig_id = int(match.group(1))
            before_img_text = s_passage[:match.start()]
            after_img_text = s_passage[match.end():]

            caption, after_caption_text = extract_image_caption(after_img_text)
            antecedent_passage = get_last_antecedent_passage(before_img_text)
            subsequent_passage = get_first_subsequent_passage(after_caption_text)

            img_record = image_records.get(fig_id)
            image_path = img_record.get("image_path") if img_record is not None else None

            image_contexts.append({
                "figure_id": fig_id,
                "caption": caption,
                "neighbor_passages": {
                    "antecedent": antecedent_passage,
                    "subsequent": subsequent_passage,
                },
                "image_path": image_path,
            })

        return image_contexts
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "semantic_chunking_utilities" / "semantic_splitter.py"] = dedent(r'''
    """LlamaIndex semantic splitter wrapper utilities."""

    from __future__ import annotations

    from typing import List

    from llama_index.core import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding


    def build_semantic_splitter(
            model_name: str,
            device: str,
            buffer_size: int = 1,
            breakpoint_percentile_threshold: int = 95,
        ) -> SemanticSplitterNodeParser:
        
        return SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=HuggingFaceEmbedding(model_name=model_name, device=device),
        )


    def semantic_chunk_passage(
            passage: str,
            semantic_splitter: SemanticSplitterNodeParser,
        ) -> List[str]:
        document = Document(text=passage)
        nodes = semantic_splitter.get_nodes_from_documents([document])
        
        return [node.get_content().strip() for node in nodes if node.get_content().strip()]
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "semantic_chunking_utilities" / "token_audit.py"] = dedent(r'''
    """BGE tokenizer audit utilities for semantic chunks."""

    from __future__ import annotations

    from sentence_transformers import SentenceTransformer


    def count_bge_tokens(text: str, encoder: SentenceTransformer) -> int:
        tokenized = encoder.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False, # Some semantic chunks from the subsection passage will still exceed BGE's limit of 512 tokens
        )
        return len(tokenized["input_ids"])
''').strip() + "\n"

# Semantic Chunking Subsection Passages Orchestrator
FILES[RAG_DIGEST_SCRIPT_DIR / "semantic_chunking_module.py"] = dedent(r'''
    """Semantic chunking and metadata construction orchestrator."""

    from __future__ import annotations

    import gc
    from typing import Any, Dict, List, Optional, Tuple

    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm.auto import tqdm

    from config import (
        BGE_MODEL_NAME,
        SEMANTIC_BREAKPOINT_PERCENTILE_THRESHOLD,
        SEMANTIC_BUFFER_SIZE,
    )
    from semantic_chunking_utilities.image_contexts import extract_image_contexts_from_subsection
    from semantic_chunking_utilities.passage_preprocessing import prepare_text_for_semantic_chunking
    from semantic_chunking_utilities.semantic_splitter import build_semantic_splitter, semantic_chunk_passage
    from semantic_chunking_utilities.token_audit import count_bge_tokens


    def run_semantic_chunking(
            subsection_chunk: Dict[str, Dict[str, Any]],
            image_records: Dict[int, Dict[str, Any]],
        ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Run semantic chunking and metadata construction.

        This stage creates lookup tables but does not encode vectors.
        Vector indexes are filled later by encoding_indexing_module.
        """
        print("[2/4] Running semantic chunking and metadata construction")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bge_encoder = SentenceTransformer(BGE_MODEL_NAME, device=device)
        bge_max_tokens = int(bge_encoder.max_seq_length)

        semantic_splitter = build_semantic_splitter(
            model_name=BGE_MODEL_NAME,
            device=device,
            buffer_size=SEMANTIC_BUFFER_SIZE,
            breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_PERCENTILE_THRESHOLD,
        )

        image_lookup_table: Dict[str, Dict[str, Any]] = {}
        chunk_lookup_table: Dict[str, Dict[str, Any]] = {}
        subsection_metadata: Dict[str, Dict[str, Any]] = {}

        s_chunk_counter = 0
        truncated_chunk_count = 0
        current_section_title: Optional[str] = None

        for subsection_key, s_metadata in tqdm(subsection_chunk.items(), desc="Semantic chunking"):
            s_id = int(s_metadata["header_id"])
            s_sub_title = s_metadata["title"]
            s_hierarchy = int(s_metadata["hierarchy"])
            s_passage = s_metadata["passage"]

            if current_section_title is None or s_hierarchy == 1:
                current_section_title = s_sub_title

            figure_ids: List[int] = []
            image_contexts = extract_image_contexts_from_subsection(
                s_passage=s_passage,
                image_records=image_records,
            )

            for image_context in image_contexts:
                fig_id = int(image_context["figure_id"])
                image_lookup_table[str(fig_id)] = {
                    "figure_id": fig_id,
                    "caption": image_context["caption"],
                    "neighbor_passages": image_context["neighbor_passages"],
                    "image_path": image_context["image_path"],
                    "vector_index": None,
                    "subsection_key": subsection_key,
                }
                figure_ids.append(fig_id)

            chunk_ready_passage = prepare_text_for_semantic_chunking(s_passage)
            semantic_chunks = semantic_chunk_passage(chunk_ready_passage, semantic_splitter)

            chunk_ids: List[int] = []
            for chunk_passage in semantic_chunks:
                vector_index = s_chunk_counter
                s_chunk_counter += 1
                
                token_count = count_bge_tokens(chunk_passage, bge_encoder)
                exceeds_encoder_limit = token_count > bge_max_tokens
                
                if exceeds_encoder_limit:
                    truncated_chunk_count += 1
                
                chunk_lookup_table[str(vector_index)] = {
                    "vector_index": vector_index,
                    "chunk_id": s_chunk_counter,
                    "chunk_passage": chunk_passage,
                    "subsection_key": subsection_key,
                    "token_count": token_count,
                    "max_seq_length": bge_max_tokens,
                    "exceeds_encoder_limit": exceeds_encoder_limit,
                }
                chunk_ids.append(s_chunk_counter)

            subsection_metadata[str(s_id + 1)] = {
                "subsection_id": s_id + 1,
                "section_title": current_section_title,
                "subsection_title": s_sub_title,
                "subsection_passage": s_passage,
                "chunk_ids": chunk_ids,
                "figure_ids": figure_ids,
                "subsection_key": subsection_key,
            }

        total_chunks = len(chunk_lookup_table)
        truncation_percentage = (
            (truncated_chunk_count / total_chunks) * 100
            if total_chunks > 0
            else 0.0
        )

        print(f"\tImage lookup records: {len(image_lookup_table)}")
        print(f"\tChunk lookup records: {len(chunk_lookup_table)}")
        print(f"\tSubsection metadata records: {len(subsection_metadata)}")
        print(
            f"\tChunks Truncation Ratio: "
            f"{truncated_chunk_count} / {total_chunks} "
            f"({truncation_percentage:.2f}%)"
        )
        
        del semantic_splitter
        del bge_encoder

        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return image_lookup_table, chunk_lookup_table, subsection_metadata
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "indexing_utilities" / "bge_encoder.py"] = dedent(r'''
    """BGE text encoding utilities."""

    from __future__ import annotations

    import numpy as np
    from sentence_transformers import SentenceTransformer


    def load_bge_encoder(model_name: str, device: str) -> SentenceTransformer:
        return SentenceTransformer(model_name, device=device)


    def encode_text_bge(text: str, bge_encoder: SentenceTransformer) -> np.ndarray:
        embedding = bge_encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return embedding.astype("float32")
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "indexing_utilities" / "clip_encoder.py"] = dedent(r'''
    """CLIP image encoding utilities."""

    from __future__ import annotations

    from pathlib import Path
    from typing import Optional, Tuple

    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor


    def load_clip_model(model_name: str, device: str) -> Tuple[CLIPModel, CLIPProcessor]:
        clip_model = CLIPModel.from_pretrained(model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_name)
        clip_model.eval()
        return clip_model, clip_processor


    @torch.no_grad()
    def encode_image_clip(
            image_path: str,
            clip_model: CLIPModel,
            clip_processor: CLIPProcessor,
            device: str,
        ) -> Optional[np.ndarray]:
        """Encode one image using CLIP ViT-B/32 and return a normalized float32 vector."""
        if image_path is None:
            print("[CLIP SKIP] image_path is None")
            return None

        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"[CLIP SKIP] path does not exist: {image_path_obj}")
            return None

        inputs = clip_processor(
            images=Image.open(image_path_obj).convert("RGB"),
            return_tensors="pt",
        )

        vision_outputs = clip_model.vision_model(
            pixel_values=inputs["pixel_values"].to(device),
            return_dict=True,
        )

        image_features = clip_model.visual_projection(vision_outputs.pooler_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze(0).detach().cpu().numpy().astype("float32")
''').strip() + "\n"

FILES[RAG_DIGEST_SCRIPT_DIR / "indexing_utilities" / "faiss_indexing.py"] = dedent(r'''
    """FAISS index construction utilities."""

    from __future__ import annotations

    from typing import List, Optional, Tuple

    import faiss
    import numpy as np


    def build_faiss_index(embeddings: List[np.ndarray]) -> Tuple[Optional[faiss.IndexFlatIP], np.ndarray]:
        """
        Build an inner-product FAISS index.

        Inner product behaves like cosine similarity because vectors are normalized.
        """
        if not embeddings:
            return None, np.empty((0, 0), dtype="float32")

        embedding_matrix = np.vstack(embeddings).astype("float32")
        faiss.normalize_L2(embedding_matrix)

        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        return index, embedding_matrix
''').strip() + "\n"

# The Sub-orchestrator for Encoding the Chunks and Images found in each subsection
FILES[RAG_DIGEST_SCRIPT_DIR / "encoding_indexing_module.py"] = dedent(r'''
    """Encoding and FAISS indexing orchestrator."""

    from __future__ import annotations
    
    import gc
    from typing import Any, Dict, List

    import numpy as np
    import torch
    from tqdm.auto import tqdm

    from config import BGE_MODEL_NAME, CLIP_MODEL_NAME
    from indexing_utilities.bge_encoder import encode_text_bge, load_bge_encoder
    from indexing_utilities.clip_encoder import encode_image_clip, load_clip_model
    from indexing_utilities.faiss_indexing import build_faiss_index


    def run_encoding_and_indexing(
            chunk_lookup_table: Dict[str, Dict[str, Any]],
            image_lookup_table: Dict[str, Dict[str, Any]],
        ) -> Dict[str, Any]:
        """Encode text/image records and build BGE/CLIP FAISS indexes."""
        print("[3/4] Running BGE/CLIP encoding and FAISS indexing")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"      Device: {device}")

        bge_encoder = load_bge_encoder(BGE_MODEL_NAME, device=device)
        clip_model, clip_processor = load_clip_model(CLIP_MODEL_NAME, device=device)

        clip_embeddings: List[np.ndarray] = []
        bge_embeddings: List[np.ndarray] = []

        image_lookup_table_by_vector_index: Dict[str, Dict[str, Any]] = {}

        for fig_id_key, image_meta in tqdm(image_lookup_table.items(), desc="CLIP image encoding"):
            image_path = image_meta.get("image_path")

            clip_vector = encode_image_clip(
                image_path=image_path,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
            )

            if clip_vector is None:
                image_meta["vector_index"] = None
                continue

            vector_index = len(clip_embeddings)
            clip_embeddings.append(clip_vector)

            image_meta["vector_index"] = vector_index

            image_lookup_table_by_vector_index[str(vector_index)] = {
                **image_meta,
                "vector_index": vector_index,
            }

        for vector_index_key, chunk_meta in tqdm(chunk_lookup_table.items(), desc="BGE text encoding"):
            vector_index = int(vector_index_key)
            chunk_passage = chunk_meta["chunk_passage"]

            expected_vector_index = len(bge_embeddings)

            if vector_index != expected_vector_index:
                raise ValueError(
                    f"Vector index mismatch: lookup key={vector_index}, "
                    f"FAISS row={expected_vector_index}"
                )

            bge_vector = encode_text_bge(chunk_passage, bge_encoder)
            bge_embeddings.append(bge_vector)

            chunk_meta["vector_index"] = vector_index

        clip_faiss_index, clip_emb_mat = build_faiss_index(clip_embeddings)
        bge_faiss_index, bge_emb_mat = build_faiss_index(bge_embeddings)

        print(f"\tCLIP embedding rows:  {len(clip_embeddings)}")
        print(f"\tBGE embedding rows:   {len(bge_embeddings)}")
        print(f"\tCLIP matrix shape:   {clip_emb_mat.shape}")
        print(f"\tBGE matrix shape:    {bge_emb_mat.shape}")
	
        del bge_encoder
        del clip_model
        del clip_processor
	
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return {
            "chunk_lookup_table": chunk_lookup_table,
            "image_lookup_table": image_lookup_table_by_vector_index,
            "bge_faiss_index": bge_faiss_index,
            "clip_faiss_index": clip_faiss_index,
            "bge_emb_mat": bge_emb_mat,
            "clip_emb_mat": clip_emb_mat,
        }
''').strip() + "\n"

# Saving the Artifacts
FILES[RAG_DIGEST_SCRIPT_DIR / "indexing_utilities" / "artifact_io.py"] = dedent(r'''
    """Artifact export utilities for JSON, CSV, NumPy, and FAISS files."""

    from __future__ import annotations

    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any, Dict

    import faiss
    import numpy as np
    import pandas as pd

    from config import (
        BGE_MODEL_NAME,
        CLIP_MODEL_NAME,
        EMBEDDING_MATRICES_DIR,
        FAISS_INDEXES_DIR,
        LOOKUP_TABLES_DIR,
        METADATA_DIR,
        REPORTS_DIR,
    )


    def json_safe(obj: Any) -> Any:
        """Recursively convert common non-JSON-safe objects into JSON-safe objects."""
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


    def write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


    def write_faiss_index(path: Path, index: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if index is not None:
            faiss.write_index(index, str(path))


    def export_rag_digest_artifacts(
            subsection_chunk: Dict[str, Dict[str, Any]],
            image_records: Dict[int, Dict[str, Any]],
            table_records: Dict[int, Dict[str, Any]],
            chunk_lookup_table: Dict[str, Dict[str, Any]],
            image_lookup_table: Dict[str, Dict[str, Any]],
            subsection_metadata: Dict[str, Dict[str, Any]],
            encoding_outputs: Dict[str, Any],
        ) -> None:
        """Export all canonical RAG digest artifacts."""
        print("[4/4] Exporting RAG digest artifacts")

        write_json(LOOKUP_TABLES_DIR / "chunk_lookup_table.json", chunk_lookup_table)
        write_json(LOOKUP_TABLES_DIR / "image_lookup_table.json", image_lookup_table)

        write_json(METADATA_DIR / "subsection_metadata.json", subsection_metadata)
        write_json(METADATA_DIR / "subsection_chunk.json", subsection_chunk)
        write_json(METADATA_DIR / "image_records.json", image_records)
        write_json(METADATA_DIR / "table_records.json", table_records)

        np.save(EMBEDDING_MATRICES_DIR / "bge_embedding_matrix.npy", encoding_outputs["bge_emb_mat"])
        np.save(EMBEDDING_MATRICES_DIR / "clip_embedding_matrix.npy", encoding_outputs["clip_emb_mat"])

        write_faiss_index(FAISS_INDEXES_DIR / "bge_text.index", encoding_outputs["bge_faiss_index"])
        write_faiss_index(FAISS_INDEXES_DIR / "clip_image.index", encoding_outputs["clip_faiss_index"])

        subsection_audit = pd.DataFrame.from_records([
            {
                "subsection_key": key,
                "header_id": value.get("header_id"),
                "hierarchy": value.get("hierarchy"),
                "title": value.get("title"),
                "page_number": value.get("page_number"),
                "char_count": value.get("char_count"),
                "image_marker_count": value.get("image_marker_count"),
                "table_marker_count": value.get("table_marker_count"),
            }
            for key, value in subsection_chunk.items()
        ])
        subsection_audit.to_csv(REPORTS_DIR / "subsection_audit.csv", index=False)

        chunk_token_audit = pd.DataFrame.from_records([
            {
                "chunk_id": value.get("chunk_id"),
                "subsection_key": value.get("subsection_key"),
                "vector_index": value.get("vector_index"),
                "token_count": value.get("token_count"),
                "max_seq_length": value.get("max_seq_length"),
                "exceeds_encoder_limit": value.get("exceeds_encoder_limit"),
                "char_count": len(value.get("chunk_passage", "")),
            }
            for value in chunk_lookup_table.values()
        ])
        chunk_token_audit.to_csv(REPORTS_DIR / "chunk_token_audit.csv", index=False)

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bge_model_name": BGE_MODEL_NAME,
            "clip_model_name": CLIP_MODEL_NAME,
            "subsection_count": len(subsection_chunk),
            "image_record_count": len(image_records),
            "table_record_count": len(table_records),
            "chunk_lookup_count": len(chunk_lookup_table),
            "image_lookup_count": len(image_lookup_table),
            "bge_embedding_matrix_shape": list(encoding_outputs["bge_emb_mat"].shape),
            "clip_embedding_matrix_shape": list(encoding_outputs["clip_emb_mat"].shape),
            "bge_faiss_rows": int(encoding_outputs["bge_faiss_index"].ntotal) if encoding_outputs["bge_faiss_index"] is not None else 0,
            "clip_faiss_rows": int(encoding_outputs["clip_faiss_index"].ntotal) if encoding_outputs["clip_faiss_index"] is not None else 0,
        }
        write_json(REPORTS_DIR / "build_manifest.json", manifest)

        print(f"\tWrote lookup tables to:      {LOOKUP_TABLES_DIR}")
        print(f"\tWrote metadata to:           {METADATA_DIR}")
        print(f"\tWrote FAISS indexes to:      {FAISS_INDEXES_DIR}")
        print(f"\tWrote embedding matrices to: {EMBEDDING_MATRICES_DIR}")
        print(f"\tWrote reports to:            {REPORTS_DIR}")
''').strip() + "\n"

# Hardware Utilitzation and Performance during Build-time
FILES[RAG_DIGEST_SCRIPT_DIR / "indexing_utilities" / "performance_monitor.py"] = dedent(r'''
    """Build-time performance monitoring utilities."""

    from __future__ import annotations

    import json
    import os
    import threading
    import time
    from contextlib import contextmanager
    from pathlib import Path
    from typing import Any, Dict, Optional

    import psutil
    import torch


    def bytes_to_mb(value: int | float) -> float:
        return float(value) / (1024 ** 2)


    def directory_size_bytes(path: Path) -> int:
        total = 0
        path = Path(path)

        if not path.exists():
            return 0

        for file_path in path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size

        return total


    class BuildTimePerformanceMonitor:
        """Sample CPU RAM and GPU VRAM during the full build-time RAG digest process."""

        def __init__(self, sample_interval_seconds: float = 0.25) -> None:
            self.sample_interval_seconds = sample_interval_seconds
            self.process = psutil.Process(os.getpid())

            self._stop_event = threading.Event()
            self._thread: Optional[threading.Thread] = None

            self.start_time: Optional[float] = None
            self.end_time: Optional[float] = None

            self.stage_times_seconds: Dict[str, float] = {}

            self.baseline_system_ram_bytes = 0
            self.baseline_process_rss_bytes = 0
            self.peak_system_ram_bytes = 0
            self.peak_process_rss_bytes = 0

            self.cuda_available = torch.cuda.is_available()
            self.baseline_vram_used_bytes = 0
            self.peak_vram_used_bytes = 0
            self.peak_torch_allocated_bytes = 0
            self.peak_torch_reserved_bytes = 0


        def _sample_once(self) -> None:
            system_ram_used = psutil.virtual_memory().used
            process_rss = self.process.memory_info().rss

            self.peak_system_ram_bytes = max(self.peak_system_ram_bytes, system_ram_used)
            self.peak_process_rss_bytes = max(self.peak_process_rss_bytes, process_rss)

            if self.cuda_available:
                free_vram, total_vram = torch.cuda.mem_get_info()
                used_vram = total_vram - free_vram

                self.peak_vram_used_bytes = max(self.peak_vram_used_bytes, used_vram)
                self.peak_torch_allocated_bytes = max(
                    self.peak_torch_allocated_bytes,
                    torch.cuda.memory_allocated(),
                )
                self.peak_torch_reserved_bytes = max(
                    self.peak_torch_reserved_bytes,
                    torch.cuda.memory_reserved(),
                )


        def _sampling_loop(self) -> None:
            while not self._stop_event.is_set():
                self._sample_once()
                time.sleep(self.sample_interval_seconds)


        def start(self) -> None:
            self.start_time = time.perf_counter()

            self.baseline_system_ram_bytes = psutil.virtual_memory().used
            self.baseline_process_rss_bytes = self.process.memory_info().rss

            self.peak_system_ram_bytes = self.baseline_system_ram_bytes
            self.peak_process_rss_bytes = self.baseline_process_rss_bytes

            if self.cuda_available:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                free_vram, total_vram = torch.cuda.mem_get_info()
                self.baseline_vram_used_bytes = total_vram - free_vram
                self.peak_vram_used_bytes = self.baseline_vram_used_bytes

            self._thread = threading.Thread(
                target=self._sampling_loop,
                daemon=True,
            )
            self._thread.start()


        def stop(self) -> None:
            self._sample_once()
            self.end_time = time.perf_counter()

            self._stop_event.set()

            if self._thread is not None:
                self._thread.join(timeout=2.0)


        @contextmanager
        def stage(self, stage_name: str):
            stage_start = time.perf_counter()

            try:
                yield

            finally:
                stage_end = time.perf_counter()
                self.stage_times_seconds[stage_name] = stage_end - stage_start


        def build_report(
                self,
                artifact_root: Path,
                output_counts: Dict[str, Any],
            ) -> Dict[str, Any]:
            total_build_time_seconds = (
                self.end_time - self.start_time
                if self.start_time is not None and self.end_time is not None
                else None
            )

            artifact_size_bytes = directory_size_bytes(artifact_root)

            report = {
                "total_build_time_seconds": total_build_time_seconds,
                "stage_times_seconds": self.stage_times_seconds,

                "ram": {
                    "baseline_system_ram_mb": bytes_to_mb(self.baseline_system_ram_bytes),
                    "peak_system_ram_mb": bytes_to_mb(self.peak_system_ram_bytes),
                    "peak_system_ram_delta_mb": bytes_to_mb(
                        self.peak_system_ram_bytes - self.baseline_system_ram_bytes
                    ),

                    "baseline_process_rss_mb": bytes_to_mb(self.baseline_process_rss_bytes),
                    "peak_process_rss_mb": bytes_to_mb(self.peak_process_rss_bytes),
                    "peak_process_rss_delta_mb": bytes_to_mb(
                        self.peak_process_rss_bytes - self.baseline_process_rss_bytes
                    ),
                },

                "vram": {
                    "cuda_available": self.cuda_available,
                    "baseline_vram_used_mb": bytes_to_mb(self.baseline_vram_used_bytes),
                    "peak_vram_used_mb": bytes_to_mb(self.peak_vram_used_bytes),
                    "peak_vram_delta_mb": bytes_to_mb(
                        self.peak_vram_used_bytes - self.baseline_vram_used_bytes
                    ),
                    "peak_torch_allocated_mb": bytes_to_mb(self.peak_torch_allocated_bytes),
                    "peak_torch_reserved_mb": bytes_to_mb(self.peak_torch_reserved_bytes),
                },

                "artifacts": {
                    "artifact_root": str(artifact_root),
                    "artifact_size_mb": bytes_to_mb(artifact_size_bytes),
                },

                "output_counts": output_counts,
            }

            return report


        def write_report(
                self,
                report_path: Path,
                artifact_root: Path,
                output_counts: Dict[str, Any],
            ) -> Dict[str, Any]:
            report = self.build_report(
                artifact_root=artifact_root,
                output_counts=output_counts,
            )

            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            report_path.write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

            return report
''').strip() + "\n"

# Visualize the /report
FILES[RAG_DIGEST_SCRIPT_DIR / "indexing_utilities" / "audit_visualization.py"] = dedent(r'''
    """Audit visualization utilities for build-time RAG digest reports."""

    from __future__ import annotations

    from pathlib import Path
    from typing import Dict

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from config import REPORTS_DIR


    def _make_truncation_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int, int, float]:
        """Create stable True/False labels with class counts for hue legends."""
        df = df.copy()

        df["exceeds_encoder_limit"] = df["exceeds_encoder_limit"].astype(bool)

        true_count = int(df["exceeds_encoder_limit"].sum())
        total_count = int(len(df))
        false_count = total_count - true_count

        true_label = f"True\n({true_count})"
        false_label = f"False\n({false_count})"

        df["truncation_category"] = df["exceeds_encoder_limit"].map({
            False: false_label,
            True: true_label,
        })

        hue_order = [false_label, true_label]
        truncation_rate = (true_count / total_count * 100.0) if total_count > 0 else 0.0

        return df, hue_order, true_count, total_count, truncation_rate


    def _plot_chunk_barplot_with_count_panel(
            chunk_token_audit: pd.DataFrame,
            value_column: str,
            y_label: str,
            output_path: Path,
        ) -> None:
        """Save chunk-wise token/character bar plot with truncation-count panel."""
        plot_df = chunk_token_audit[
            ["chunk_id", value_column, "max_seq_length", "exceeds_encoder_limit"]
        ].copy()

        plot_df["chunk_id"] = plot_df["chunk_id"].astype(int)
        plot_df[value_column] = plot_df[value_column].astype(int)
        plot_df = plot_df.sort_values("chunk_id")

        plot_df, hue_order, true_count, total_count, truncation_rate = _make_truncation_labels(plot_df)

        count_df = (
            plot_df["truncation_category"]
            .value_counts()
            .reindex(hue_order)
            .reset_index()
        )
        count_df.columns = ["truncation_category", "count"]

        sns.set_theme(style="whitegrid")

        fig, (ax_bar, ax_count) = plt.subplots(
            ncols=2,
            figsize=(20, 6),
            gridspec_kw={
                "width_ratios": [11.75, 1.0],
            },
        )

        sns.barplot(
            data=plot_df,
            x="chunk_id",
            y=value_column,
            hue="truncation_category",
            hue_order=hue_order,
            dodge=False,
            errorbar=None,
            ax=ax_bar,
        )

        if value_column == "token_count":
            token_count_limit = int(plot_df["max_seq_length"].iloc[0])

            ax_bar.axhline(
                y=token_count_limit,
                linestyle="--",
                linewidth=1.5,
            )
            ax_bar.text(
                x=len(plot_df) - 1,
                y=token_count_limit,
                s=f"limit={token_count_limit}",
                va="bottom",
                ha="right",
            )

        ax_bar.set_title(f"Truncation Rate: {truncation_rate:.2f}%")
        ax_bar.set_xlabel("Chunk ID")
        ax_bar.set_ylabel(y_label)

        tick_step = max(1, len(plot_df) // 25)
        for idx, tick_label in enumerate(ax_bar.get_xticklabels()):
            if idx % tick_step != 0:
                tick_label.set_visible(False)

        sns.barplot(
            data=count_df,
            x="truncation_category",
            y="count",
            hue="truncation_category",
            hue_order=hue_order,
            order=hue_order,
            dodge=False,
            errorbar=None,
            legend=False,
            ax=ax_count,
        )

        ax_count.set_title("Count")
        ax_count.set_xlabel("")
        ax_count.set_ylabel("")
        ax_count.tick_params(axis="x", rotation=90)

        for container in ax_count.containers:
            ax_count.bar_label(container, fmt="%d", padding=2)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    def _plot_subsection_density(
            subsection_audit: pd.DataFrame,
            output_path: Path,
        ) -> None:
        """
        Save document-density plot by subsection sequence.

        Note:
            This uses seaborn.barplot, not seaborn.countplot, because the y-axis
            represents char_count. countplot would count rows rather than plot
            subsection character density.
        """
        plot_df = subsection_audit.copy()

        plot_df["header_id"] = pd.to_numeric(plot_df["header_id"], errors="coerce")
        plot_df["char_count"] = pd.to_numeric(plot_df["char_count"], errors="coerce")
        plot_df["hierarchy"] = plot_df["hierarchy"].astype(str)

        plot_df = plot_df.dropna(subset=["header_id", "char_count"])
        plot_df = plot_df.sort_values("header_id").reset_index(drop=True)

        plot_df["subsection_sequence"] = plot_df.index + 1
        plot_df["hierarchy_category"] = "Hierarchy " + plot_df["hierarchy"]

        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots(figsize=(18, 6))

        sns.barplot(
            data=plot_df,
            x="subsection_sequence",
            y="char_count",
            hue="hierarchy_category",
            dodge=False,
            errorbar=None,
            ax=ax,
        )

        ax.set_title("Document density")
        ax.set_xlabel("Subsection (Sequential)")
        ax.set_ylabel("Character Count of the Subsection")

        # Keep x-axis readable for 105 subsections.
        tick_step = max(1, len(plot_df) // 20)
        for idx, tick_label in enumerate(ax.get_xticklabels()):
            if idx % tick_step != 0:
                tick_label.set_visible(False)

        ax.legend(title="Hierarchy", loc="upper right")

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    def visualize_rag_digest_reports(
            reports_dir: Path = REPORTS_DIR,
        ) -> Dict[str, str]:
        """
        Read exported audit CSV files and save visualization PNGs under reports_dir.

        Returns:
            Dictionary of plot names to saved file paths.
        """
        reports_dir = Path(reports_dir)

        chunk_token_audit_path = reports_dir / "chunk_token_audit.csv"
        subsection_audit_path = reports_dir / "subsection_audit.csv"

        chunk_token_audit = pd.read_csv(chunk_token_audit_path)
        subsection_audit = pd.read_csv(subsection_audit_path)

        token_plot_path = reports_dir / "visualization_chunk_token_count.png"
        char_plot_path = reports_dir / "visualization_chunk_character_count.png"
        density_plot_path = reports_dir / "visualization_subsection_document_density.png"

        _plot_chunk_barplot_with_count_panel(
            chunk_token_audit=chunk_token_audit,
            value_column="token_count",
            y_label="Token Count",
            output_path=token_plot_path,
        )

        _plot_chunk_barplot_with_count_panel(
            chunk_token_audit=chunk_token_audit,
            value_column="char_count",
            y_label="Character Count",
            output_path=char_plot_path,
        )

        _plot_subsection_density(
            subsection_audit=subsection_audit,
            output_path=density_plot_path,
        )

        return {
            "chunk_token_count_histogram": str(token_plot_path),
            "chunk_character_count_histogram": str(char_plot_path),
            "subsection_document_density": str(density_plot_path),
        }
''').strip() + "\n"

# Main Orchestrator
FILES[RAG_DIGEST_SCRIPT_EXEC_PY_PATH] = dedent(r'''
    """Main build-time RAG digest orchestrator."""

    from __future__ import annotations

    import sys
    from pathlib import Path


    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    from config import ARTIFACTS_DIR, PDF_PATH, REPORTS_DIR
    from encoding_indexing_module import run_encoding_and_indexing
    from indexing_utilities.artifact_io import export_rag_digest_artifacts
    from indexing_utilities.performance_monitor import BuildTimePerformanceMonitor
    from indexing_utilities.audit_visualization import visualize_rag_digest_reports
    from semantic_chunking_module import run_semantic_chunking
    from subsection_chunking_module import run_subsection_chunking


    def main() -> None:
        monitor = BuildTimePerformanceMonitor(sample_interval_seconds=0.25)
        monitor.start()
        
        print("Running RAG digest build-time pipeline")
        print(f"PDF path: {PDF_PATH}")
        print(f"Artifacts directory: {ARTIFACTS_DIR}")

        try:
            print("-"*100)
            with monitor.stage("subsection_chunking"):
                subsection_chunk, image_records, table_records = run_subsection_chunking(
                    pdf_path=PDF_PATH,
                )
                
            print("-"*100)
            with monitor.stage("semantic_chunking_metadata"):
                image_lookup_table, chunk_lookup_table, subsection_metadata = run_semantic_chunking(
                    subsection_chunk=subsection_chunk,
                    image_records=image_records,
                )
        
            print("-"*100)
            with monitor.stage("encoding_indexing"):
                encoding_outputs = run_encoding_and_indexing(
                    chunk_lookup_table=chunk_lookup_table,
                    image_lookup_table=image_lookup_table,
                )
        
            chunk_lookup_table = encoding_outputs["chunk_lookup_table"]
            image_lookup_table = encoding_outputs["image_lookup_table"]
        
            print("-"*100)
            with monitor.stage("artifact_export_visualization"):
                export_rag_digest_artifacts(
                    subsection_chunk=subsection_chunk,
                    image_records=image_records,
                    table_records=table_records,
                    chunk_lookup_table=chunk_lookup_table,
                    image_lookup_table=image_lookup_table,
                    subsection_metadata=subsection_metadata,
                    encoding_outputs=encoding_outputs,
                )
        
                visualize_rag_digest_reports(
                    reports_dir=REPORTS_DIR,
                )
        finally:
            monitor.stop()

            monitor.write_report(
                report_path=REPORTS_DIR / "build_time_performance_report.json",
                artifact_root=ARTIFACTS_DIR,
                output_counts={
                    "subsection_count": len(subsection_chunk) if "subsection_chunk" in locals() else None,
                    "image_record_count": len(image_records) if "image_records" in locals() else None,
                    "table_record_count": len(table_records) if "table_records" in locals() else None,
                    "chunk_lookup_count": len(chunk_lookup_table) if "chunk_lookup_table" in locals() else None,
                    "image_lookup_count": len(encoding_outputs["image_lookup_table"]) if "encoding_outputs" in locals() else None,
                    "bge_embedding_rows": int(encoding_outputs["bge_emb_mat"].shape[0]) if "encoding_outputs" in locals() else None,
                    "clip_embedding_rows": int(encoding_outputs["clip_emb_mat"].shape[0]) if "encoding_outputs" in locals() else None,
                }
            )
        
        print("-"*100)
        print("[INFO] RAG digest artifacts generated successfully!")


    if __name__ == "__main__":
        main()
''').strip() + "\n"

def write_file(path: Path, content: str) -> None:
    if path.exists() and not OVERWRITE_MODULES:
        print(f"[INFO][SKIP] file exists: {path.relative_to(PROJECT_ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    action = "OVERWRITE" if path.exists() else "CREATE"
    print(f"[INFO][{action}] file: {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")

    if RESET_BUILD_TIME_DIR and BUILD_TIME_DIR.exists():
        print("[INFO] Resetting .../build_time. You can disable this behavior by setting `RESET_BUILD_TIME` to `False` in the bootstrap for build-time.")
        shutil.rmtree(BUILD_TIME_DIR)
    if OVERWRITE_MODULES and not RESET_BUILD_TIME_DIR:
        print("[INFO] Existing modules are overwritten. You can disable this behavior by setting `OVERWRITE_MODULES` to `False`")

    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] directory: {directory.relative_to(PROJECT_ROOT)}")
    print("="*100)

    for path, content in FILES.items():
        write_file(path, content)
    print("="*100)
    
    subprocess.run(
        [sys.executable, str(RAG_DIGEST_SCRIPT_EXEC_PY_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("\nBuild-time implementation files are populated.")


if __name__ == "__main__":
    main()

