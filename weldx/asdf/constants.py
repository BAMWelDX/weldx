from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas/weldx.bam.de/weldx"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests"

WELDX_URI_BASE = "asdf://weldx.bam.de/weldx/"
WELDX_TAG_BASE = "asdf://weldx.bam.de/weldx/tags/"
WELDX_EXTENSION_URI_BASE = "asdf://weldx.bam.de/weldx/extensions/weldx"
