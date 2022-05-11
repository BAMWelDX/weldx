# clean up jupyter notebooks in main directory
from __future__ import annotations

import json
from typing import Union


def _clean_notebook(file: Union[str, Path]):  # pragma: no cover
    """Clean ID metadata, output and execution count from jupyter notebook cells.

    This function overrides the existing notebook file, use with caution!

    Parameters
    ----------
    file :
        The jupyter notebook filename to clean.

    """
    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    for cell in data["cells"]:
        cell.pop("id", None)
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None

    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    from pathlib import Path

    p = Path(__file__).resolve().parent
    notebooks = p.glob("../../tutorials/*.ipynb")
    for nb in notebooks:
        _clean_notebook(nb)
