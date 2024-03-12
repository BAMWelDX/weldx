"""Common util functions used in weldx tutorials."""

from pathlib import Path

tutorials_dir = Path(__file__).parent.absolute()


def download_tutorial_input_file(print_status=True):
    from urllib.request import urlretrieve

    url = (
        "https://github.com/BAMWelDX/IIW2021_AA_CXII/blob/weldx_0.6.0"
        "/single_pass_weld.weldx?raw=true "
    )
    sha256sum = "d3329721e1eeb73fcaac2d5f5f649aa267d5c705abbc314bb31e53147e080297"

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
            if print_status:
                print(f"File {dest} already downloaded.")
            return

    # does not exist or hash mismatched, so download it.
    if print_status:
        print("trying to download: {url}")
    out_file, _ = urlretrieve(url, dest)  # skipcq: BAN-B310
    sha256sum_actual = hash_path(out_file)
    if sha256sum_actual != sha256sum:
        raise RuntimeError(
            f"hash mismatch:\n actual = \t{sha256sum_actual}\n"
            f"desired = \t{sha256sum}"
        )
    if print_status:
        print("download successful.")
