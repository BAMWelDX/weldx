# clean up jupyter notebooks in main directory
if __name__ == "__main__":
    import subprocess
    from pathlib import Path

    files = Path("./../../weldx/").rglob("*.py")
    for p in files:
        subprocess.run(  # skipcq: BAN-B607
            ["pyupgrade", str(p.resolve()), "--py38-plus", "--keep-runtime-typing"],
            check=True,
        )
