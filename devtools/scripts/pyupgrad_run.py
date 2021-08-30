import subprocess
from pathlib import Path

files = Path("./../../weldx/").rglob("*.py")
for p in files:
    subprocess.run(
        ["pyupgrade", str(p.resolve()), "--py38-plus", "--keep-runtime-typing"]
    )
