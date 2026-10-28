"""Checagem do dataset bruto: classes, contagens, arquivos corrompidos e duplicatas.

Gera:
- data/processed/manifest.csv  (uma linha por arquivo; usado depois no split)
- docs/dados.md                (relatório legível)

Uso:
    python -m pokedex.data.check
"""

import csv
import hashlib
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from pokedex.config import data_path, data_root, load_config, project_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    """Informações de um arquivo do dataset."""

    path: str
    label: str
    extension: str
    ok: bool
    error: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    md5: str = ""
    phash: str = ""
    dup_group: int = -1


def find_dataset_root(raw_dir: Path) -> Path:
    """Acha a pasta cujas subpastas são as classes (cada uma com imagens dentro)."""
    candidates = [raw_dir, *(p for p in raw_dir.rglob("*") if p.is_dir())]
    for directory in candidates:
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        with_images = [
            d for d in subdirs if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir())
        ]
        if len(with_images) >= 2:
            return directory
    raise FileNotFoundError(f"Nenhuma pasta de classes encontrada em {raw_dir}")


def list_files(root: Path) -> list[tuple[str, Path]]:
    """Lista (classe, arquivo) para todos os arquivos dentro das pastas de classe."""
    return [
        (class_dir.name, f)
        for class_dir in sorted(d for d in root.iterdir() if d.is_dir())
        for f in sorted(class_dir.rglob("*"))
        if f.is_file()
    ]


def file_md5(path: Path) -> str:
    """MD5 do conteúdo do arquivo (detecta cópias byte a byte)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def inspect_image(path: Path, label: str, root: Path) -> ImageRecord:
    """Abre a imagem, valida e calcula hashes. Nunca levanta exceção."""
    record = ImageRecord(
        path=path.relative_to(root).as_posix(), label=label, extension=path.suffix.lower(), ok=False
    )
    if record.extension not in IMAGE_EXTENSIONS:
        record.error = "extensão não suportada"
        return record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as img:
                img.verify()  # checa a estrutura do arquivo
            with Image.open(path) as img:  # verify() invalida o objeto; reabre
                img.load()  # decodifica de fato (pega arquivos truncados)
                record.width, record.height = img.size
                record.mode = img.mode
                record.phash = str(imagehash.phash(img.convert("RGB")))
        record.md5 = file_md5(path)
        record.ok = True
    except Exception as exc:  # qualquer falha de leitura = arquivo corrompido
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def _popcount(x: np.ndarray) -> np.ndarray:
    """Número de bits 1 em cada uint64."""
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(axis=-1)


def find_duplicate_groups(phashes: list[str], max_distance: int, chunk: int = 512) -> list[int]:
    """Agrupa hashes com distância de Hamming <= max_distance (union-find).

    Retorna um id de grupo por hash; imagens no mesmo grupo são (quase) duplicatas.
    """
    n = len(phashes)
    values = np.array([int(h, 16) for h in phashes], dtype=np.uint64)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, chunk):
        block = values[start : start + chunk]
        distances = _popcount(block[:, None] ^ values[None, :])
        rows, cols = np.nonzero(distances <= max_distance)
        for r, c in zip(rows + start, cols, strict=True):
            if c > r:
                parent[find(int(c))] = find(int(r))

    roots = [find(i) for i in range(n)]
    ids = {root: idx for idx, root in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def write_manifest(records: list[ImageRecord], path: Path) -> None:
    """Salva os registros em CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)


def build_report(records: list[ImageRecord], dataset_root: Path, cfg: dict) -> str:
    """Monta o relatório em Markdown."""
    ok = [r for r in records if r.ok]
    bad = [r for r in records if not r.ok]
    counts = Counter(r.label for r in ok)
    all_labels = sorted({r.label for r in records})
    values = sorted(counts.values())
    median = statistics.median(values)

    groups: dict[int, list[ImageRecord]] = defaultdict(list)
    for r in ok:
        groups[r.dup_group].append(r)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    cross_class = [g for g in dup_groups if len({r.label for r in g}) > 1]
    md5_counts = Counter(r.md5 for r in ok)
    exact_extra = sum(c - 1 for c in md5_counts.values() if c > 1)
    near_extra = sum(len(g) - 1 for g in dup_groups)

    small = [r for r in ok if min(r.width, r.height) < cfg["min_side_px"]]
    widths = [r.width for r in ok]
    heights = [r.height for r in ok]
    min_label = min(counts, key=counts.get)
    max_label = max(counts, key=counts.get)

    lines = [
        "# Relatório de dados",
        "",
        f"Gerado por `python -m pokedex.data.check` em {date.today().isoformat()}.",
        f"Dataset: `{cfg['dataset_handle']}` — pasta `{dataset_root.relative_to(data_root(cfg)).as_posix()}`.",
        "",
        "## Resumo",
        "",
        "| Item | Valor |",
        "|---|---|",
        f"| Classes (pastas) | {len(all_labels)} |",
        f"| Arquivos | {len(records)} |",
        f"| Imagens válidas | {len(ok)} |",
        f"| Arquivos com problema | {len(bad)} |",
        f"| Cópias exatas (MD5) além da 1ª | {exact_extra} |",
        f"| Grupos de (quase) duplicatas, pHash ≤ {cfg['near_duplicate_max_distance']} | {len(dup_groups)} ({near_extra} imagens a mais) |",
        f"| Grupos com classes diferentes | {len(cross_class)} |",
        f"| Imagens com lado < {cfg['min_side_px']} px | {len(small)} |",
        "",
        "## Desbalanceamento",
        "",
        f"- Mínimo: **{counts[min_label]}** ({min_label}) · Máximo: **{counts[max_label]}** ({max_label})",
        f"- Média: {statistics.mean(values):.1f} · Mediana: {median:.0f} · Desvio padrão: {statistics.pstdev(values):.1f}",
        f"- Razão máx/mín: **{counts[max_label] / counts[min_label]:.1f}x**",
        f"- Classes com menos da metade da mediana: {sum(v < median / 2 for v in values)}",
        "",
        "## Formatos",
        "",
        f"- Extensões: {dict(Counter(r.extension for r in records).most_common())}",
        f"- Modos de cor: {dict(Counter(r.mode for r in ok).most_common())}",
        f"- Largura: mín {min(widths)}, mediana {statistics.median(widths):.0f}, máx {max(widths)}",
        f"- Altura: mín {min(heights)}, mediana {statistics.median(heights):.0f}, máx {max(heights)}",
        "",
        "## Arquivos com problema",
        "",
    ]
    lines += [f"- `{r.path}` — {r.error}" for r in bad] or ["Nenhum."]

    lines += ["", "## Duplicatas entre classes diferentes", ""]
    if cross_class:
        lines.append("Possíveis erros de rótulo. Arquivos de cada grupo:")
        lines.append("")
        for g in cross_class:
            lines.append("- " + ", ".join(f"`{r.path}`" for r in g))
    else:
        lines.append("Nenhuma.")

    lines += ["", "## Exemplos de duplicatas na mesma classe (até 15 grupos)", ""]
    same_class = [g for g in dup_groups if g not in cross_class]
    lines += ["- " + ", ".join(f"`{r.path}`" for r in g) for g in same_class[:15]] or ["Nenhuma."]

    lines += [
        "",
        "## Imagens por classe",
        "",
        "<details><summary>Tabela completa</summary>",
        "",
        "| Classe | Imagens |",
        "|---|---|",
    ]
    lines += [f"| {label} | {counts.get(label, 0)} |" for label in all_labels]
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def run_check(cfg: dict) -> tuple[list[ImageRecord], str]:
    """Executa toda a checagem e devolve (registros, relatório)."""
    dataset_root = find_dataset_root(data_path(cfg, "raw_dir"))
    files = list_files(dataset_root)
    print(f"{len(files)} arquivos em {dataset_root}")

    records = []
    for i, (label, path) in enumerate(files, 1):
        records.append(inspect_image(path, label, dataset_root))
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")

    ok = [r for r in records if r.ok]
    for r, group in zip(
        ok,
        find_duplicate_groups([r.phash for r in ok], cfg["near_duplicate_max_distance"]),
        strict=True,
    ):
        r.dup_group = group
    return records, build_report(records, dataset_root, cfg)


def main() -> None:
    cfg = load_config("data")
    records, report = run_check(cfg)

    manifest = data_path(cfg, "processed_dir") / "manifest.csv"
    write_manifest(records, manifest)
    report_path = project_path(cfg["report_path"])
    report_path.write_text(report, encoding="utf-8")
    print(f"Manifesto: {manifest}\nRelatório: {report_path}")


if __name__ == "__main__":
    main()
