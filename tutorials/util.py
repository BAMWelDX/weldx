"""Common util functions used in weldx tutorials."""
from pathlib import Path

tutorials_dir = Path(__file__).parent.absolute()


def download_tutorial_input_file():
    from urllib.request import urlretrieve

    url = (
        "https://github.com/BAMWelDX/IIW2021_AA_CXII/blob/weldx_0.5.0"
        "/single_pass_weld.weldx?raw=true "
    )
    sha256sum = "29e4f11ef1185f818b4611860842ef52d386ad2866a2680257950f160e1e098a"

    def hash_path(path):
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()

    dest = tutorials_dir / "single_pass_weld.wx"

    # check if existing files matches desired one.
    if dest.exists():
        hash_local = hash_path(dest)
        if hash_local == sha256sum:
            print(f"File {dest} already downloaded.")
            return

    # does not exist or hash mismatched, so download it.
    print("trying to download: {url}")
    out_file, header = urlretrieve(url, dest)  # skipcq: BAN-B310
    sha256sum_actual = hash_path(out_file)
    if not sha256sum_actual == sha256sum:
        raise RuntimeError(
            f"hash mismatch:\n actual = \t{sha256sum_actual}\n"
            f"desired = \t{sha256sum}"
        )

    print("download successful.")
