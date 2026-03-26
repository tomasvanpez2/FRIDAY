import os
from pathlib import Path
import re


ALLOWED_NON_ASCII = "ñÑáéíóúÁÉÍÓÚüÜ"


def _iter_files(root: Path, exts: set[str]) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def _read_text(path: Path) -> str:
    # Prefer UTF-8; fall back to latin-1 to avoid crashing on legacy files.
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def _collapse_consecutive_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = (line.strip() == "")
        if is_blank:
            if prev_blank:
                continue
            out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    return out


def _clean_text(raw: str) -> str:
    # 1) Normalize newlines early
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 2) Remove consecutive empty lines (keep at most one)
    lines = raw.split("\n")
    lines = _collapse_consecutive_blank_lines(lines)
    raw = "\n".join(lines)

    # 3) Remove non-ASCII characters except ñ and tildes (accents)
    # Keep tabs/newlines for now; later we normalize whitespace anyway.
    raw = re.sub(rf"[^\x09\x0A\x20-\x7E{re.escape(ALLOWED_NON_ASCII)}]", " ", raw)

    # 4) Normalize spaces: collapse any whitespace run to single space
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _chunk_words(words: list[str], chunk_size: int = 400, overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]  # .../viernes
    viernes_root = repo_root
    project_root = repo_root.parent

    sources: list[tuple[str, Path, set[str]]] = [
        ("tecnico", project_root / "viernes" / "data" / "corpus" / "tecnico", {".txt", ".md"}),
        ("herramientas", project_root / "viernes" / "data" / "corpus" / "herramientas", {".txt", ".md"}),
        ("conversacion", project_root / "viernes" / "data" / "corpus" / "conversacion", {".txt", ".md"}),
        ("proyecto", project_root / "viernes" / "data" / "corpus" / "proyecto", {".txt", ".md"}),
        ("viernes", viernes_root, {".py", ".md"}),
    ]

    out_path = project_root / "viernes" / "data" / "corpus_final.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_counts: dict[str, int] = {k: 0 for (k, _, _) in sources}
    total_chunks = 0
    total_words = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as f_out:
        for key, root, exts in sources:
            for path in _iter_files(root, exts):
                # Avoid reading the output file if it already exists in viernes/
                if path.resolve() == out_path.resolve():
                    continue

                file_counts[key] += 1

                raw = _read_text(path)
                cleaned = _clean_text(raw)
                words = cleaned.split()
                chunks = _chunk_words(words, chunk_size=400, overlap=50)

                for ch in chunks:
                    f_out.write(ch + "\n")
                    total_chunks += 1
                    total_words += len(ch.split())

    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    size_mb = size_bytes / (1024 * 1024)

    print("=== VIERNES corpus preparation stats ===")
    print("Archivos procesados por carpeta:")
    for key, _, _ in sources:
        print(f"- {key}: {file_counts[key]}")
    print(f"Total de chunks generados: {total_chunks}")
    print(f"Total de palabras: {total_words}")
    print(f"Tamaño de corpus_final.txt: {size_mb:.2f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
