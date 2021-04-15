# clean up jupyter notebooks in main directory
if __name__ == "__main__":
    from pathlib import Path

    from weldx.util import _clean_notebook_ids

    p = Path(__file__).parents[1]
    notebooks = p.glob("tutorials/*.ipynb")
    for nb in notebooks:
        _clean_notebook_ids(nb)
