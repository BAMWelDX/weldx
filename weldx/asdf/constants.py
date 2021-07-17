import os
from pathlib import Path

from asdf.util import filepath_to_url

SCHEMA_PATH = str(Path(__file__).resolve().parents[0] / "schemas")
WELDX_SCHEMA_URI_BASE = "http://weldx.bam.de/schemas/"
WELDX_TAG_BASE = "asdf://weldx.bam.de/weldx/tags/"
WELDX_URL_MAPPING = [
    (
        WELDX_SCHEMA_URI_BASE,
        filepath_to_url(os.path.join(SCHEMA_PATH, "weldx.bam.de"))
        + "/{url_suffix}.yaml",
    )
]
