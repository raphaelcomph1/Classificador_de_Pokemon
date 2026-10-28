"""Split estratificado treino/validação/teste a partir do manifest.csv.

Copia as imagens mantidas para processed/{train,val,test}/<classe>/, no formato
que o `ImageFolder` do torchvision espera, com a extensão corrigida pelo formato real.

Uso:
    python -m pokedex.data.check   # antes: gera o manifesto
    python -m pokedex.data.split
"""

import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from pokedex.config import data_path, load_config, project_path
from pokedex.data.check import find_dataset_root

SPLITS = ("train", "val", "test")
EXTENSION_BY_FORMAT = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif", "BMP": ".bmp", "WEBP": ".webp"}
REPORT_MARKER = "<!-- split -->"


def read_kept_rows(manifest: Path) -> list[dict[str, str]]:
    """Lê o manifesto e devolve só as imagens mantidas (excluded == False)."""
    with manifest.open(encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["excluded"] == "False"]


def check_no_duplicates(rows: list[dict[str, str]]) -> None:
    """Garante que não sobrou nenhum grupo de duplicatas (senão haveria vazamento)."""
    repeated = [g for g, n in Counter(r["dup_group"] for r in rows).items() if n > 1]
    if repeated:
        raise ValueError(f"{len(repeated)} grupos de duplicatas ainda no dataset; rode o check.")


def assign_splits(
    rows: list[dict[str, str]], ratios: dict[str, float], seed: int
) -> dict[str, list[dict[str, str]]]:
    """Divide cada classe separadamente (estratificação) com embaralhamento por seed.

    val e test recebem round(n * proporção), com ao menos 1 imagem cada; o resto vai para train.
    """
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["path"]):  # ordem fixa antes de embaralhar
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    result: dict[str, list[dict[str, str]]] = {s: [] for s in SPLITS}
    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * ratios["val"]))
        n_test = max(1, round(len(items) * ratios["test"]))
        result["val"] += items[:n_val]
        result["test"] += items[n_val : n_val + n_test]
        result["train"] += items[n_val + n_test :]
    return result


def output_name(row: dict[str, str]) -> str:
    """Nome do arquivo de saída: md5 (único) + extensão do formato real."""
    return row["md5"][:16] + EXTENSION_BY_FORMAT[row["format"]]


def copy_splits(splits: dict[str, list[dict[str, str]]], src_root: Path, out_dir: Path) -> None:
    """Recria processed/{train,val,test}/<classe>/ copiando as imagens."""
    for split, rows in splits.items():
        split_dir = out_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)  # pasta gerada: sempre recriada do zero
        for row in rows:
            dst = split_dir / row["label"] / output_name(row)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_root / row["path"], dst)


def write_splits_csv(splits: dict[str, list[dict[str, str]]], path: Path) -> None:
    """Salva split, classe, arquivo original e arquivo copiado (rastreabilidade)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "source_path", "file"])
        for split, rows in splits.items():
            for row in sorted(rows, key=lambda r: (r["label"], r["path"])):
                writer.writerow([split, row["label"], row["path"], output_name(row)])


def build_split_report(splits: dict[str, list[dict[str, str]]], cfg: dict) -> str:
    """Seção Markdown com o resumo do split."""
    total = sum(len(rows) for rows in splits.values())
    per_label = {s: Counter(r["label"] for r in rows) for s, rows in splits.items()}
    lines = [
        REPORT_MARKER,
        "## Split",
        "",
        f"Gerado por `python -m pokedex.data.split` · seed `{cfg['seed']}` · "
        f"estratificado por classe · proporções {cfg['split']}.",
        "",
        "| Split | Imagens | % | Mín. por classe | Máx. por classe |",
        "|---|---|---|---|---|",
    ]
    for s in SPLITS:
        counts = per_label[s].values()
        lines.append(
            f"| {s} | {len(splits[s])} | {100 * len(splits[s]) / total:.1f}% "
            f"| {min(counts)} | {max(counts)} |"
        )
    lines += ["", "Lista completa em `processed/splits.csv`.", ""]
    return "\n".join(lines)


def update_report(report_path: Path, section: str) -> None:
    """Substitui (ou adiciona) a seção de split no fim do relatório."""
    text = report_path.read_text(encoding="utf-8").split(REPORT_MARKER)[0].rstrip()
    report_path.write_text(text + "\n\n" + section, encoding="utf-8")


def main() -> None:
    cfg = load_config("data")
    processed = data_path(cfg, "processed_dir")
    rows = read_kept_rows(processed / "manifest.csv")
    check_no_duplicates(rows)

    splits = assign_splits(rows, cfg["split"], cfg["seed"])
    copy_splits(splits, find_dataset_root(data_path(cfg, "raw_dir")), processed)
    write_splits_csv(splits, processed / "splits.csv")
    update_report(project_path(cfg["report_path"]), build_split_report(splits, cfg))
    for s in SPLITS:
        print(f"{s}: {len(splits[s])} imagens")


if __name__ == "__main__":
    main()
