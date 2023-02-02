from weldx.constants import WELDX_PATH

SCHEMA_PATH = WELDX_PATH / "schemas/weldx.bam.de/weldx"
MANIFEST_PATH = WELDX_PATH / "manifests"

WELDX_URI_BASE = "asdf://weldx.bam.de/weldx/"
WELDX_TAG_URI_BASE = "asdf://weldx.bam.de/weldx/tags/"
WELDX_SCHEMA_URI_BASE = "asdf://weldx.bam.de/weldx/schemas/"
WELDX_EXTENSION_URI_BASE = "asdf://weldx.bam.de/weldx/extensions/"

WELDX_EXTENSION_VERSION = "0.1.2"
WELDX_EXTENSION_URI = f"{WELDX_EXTENSION_URI_BASE}weldx-{WELDX_EXTENSION_VERSION}"

WELDX_MANIFEST_URI = WELDX_URI_BASE + "manifests/weldx-" + WELDX_EXTENSION_VERSION
