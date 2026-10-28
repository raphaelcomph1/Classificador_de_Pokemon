import pytest

from pokedex.data.split import assign_splits, check_no_duplicates, output_name

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _rows(counts: dict[str, int]) -> list[dict[str, str]]:
    rows, group = [], 0
    for label, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "path": f"{label}/{i}.jpg",
                    "label": label,
                    "dup_group": str(group),
                    "md5": f"{group:032x}",
                    "format": "JPEG",
                }
            )
            group += 1
    return rows


def test_assign_splits_is_stratified_and_disjoint() -> None:
    rows = _rows({"Pikachu": 40, "Mew": 25})
    splits = assign_splits(rows, RATIOS, seed=42)

    paths = [r["path"] for rows_ in splits.values() for r in rows_]
    assert len(paths) == len(set(paths)) == len(rows)  # sem sobreposição, nada perdido

    for label, n in {"Pikachu": 40, "Mew": 25}.items():
        val = sum(r["label"] == label for r in splits["val"])
        test = sum(r["label"] == label for r in splits["test"])
        assert val == test == round(n * 0.15)  # cada classe dividida separadamente


def test_assign_splits_is_deterministic() -> None:
    rows = _rows({"Pikachu": 40, "Mew": 25})
    a = assign_splits(rows, RATIOS, seed=42)
    b = assign_splits(list(reversed(rows)), RATIOS, seed=42)  # ordem de entrada não importa
    c = assign_splits(rows, RATIOS, seed=7)
    assert [r["path"] for r in a["test"]] == [r["path"] for r in b["test"]]
    assert [r["path"] for r in a["test"]] != [r["path"] for r in c["test"]]


def test_small_class_gets_at_least_one_val_and_test() -> None:
    splits = assign_splits(_rows({"Ditto": 3}), RATIOS, seed=0)
    assert len(splits["val"]) == len(splits["test"]) == 1


def test_check_no_duplicates_raises_on_shared_group() -> None:
    rows = _rows({"Pikachu": 2})
    rows[1]["dup_group"] = rows[0]["dup_group"]
    with pytest.raises(ValueError):
        check_no_duplicates(rows)


def test_output_name_uses_real_format() -> None:
    row = {"md5": "ab" * 16, "format": "PNG"}
    assert output_name(row) == "abababababababab.png"
