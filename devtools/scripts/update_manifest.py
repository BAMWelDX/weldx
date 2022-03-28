"""Update the manifest file with schema files found in this directory."""
from pathlib import Path

import yaml

from weldx.asdf.constants import (
    WELDX_EXTENSION_URI,
    WELDX_EXTENSION_VERSION,
    WELDX_MANIFEST_URI,
)
from weldx.asdf.util import get_converter_for_tag


def update_manifest(
    search_dir: str = "../../weldx/schemas",
    out: str = f"../../weldx/manifests/weldx-{WELDX_EXTENSION_VERSION}.yaml",
):
    """Create manifest file from existing schemas."""
    # read existing manifest
    manifest = yaml.load(
        Path(out).read_text(),
        Loader=yaml.SafeLoader,
    )

    manifest["id"] = WELDX_MANIFEST_URI
    manifest["extension_uri"] = WELDX_EXTENSION_URI

    # keep only ASDF schema mappings
    manifest["tags"] = [
        mapping
        for mapping in manifest["tags"]
        if mapping["schema_uri"].startswith("http://stsci.edu/schemas")
    ]

    schemas = sorted(Path(search_dir).rglob("*.yaml"))

    for schema in schemas:
        content = yaml.load(
            schema.read_text(),
            Loader=yaml.SafeLoader,
        )
        if "id" in content:  # should be schema file
            uri: str = content["id"]
            if uri.startswith("asdf://weldx.bam.de"):
                tag = uri.replace("/schemas/", "/tags/")
            else:
                raise ValueError(f"Unknown URI format {uri=}")

            if tag is not None and get_converter_for_tag(
                tag
            ):  # check if converter is implemented
                manifest["tags"].append(dict(tag_uri=tag, schema_uri=uri))
            else:
                print(f"No converter for URI: {schema}")

    with open(Path(out), "w") as outfile:  # skipcq: PTC-W6004
        outfile.write("%YAML 1.1\n---\n")
        yaml.dump(
            manifest,
            outfile,
            default_flow_style=False,
            sort_keys=False,
        )
        outfile.write("...\n")


if __name__ == "__main__":
    update_manifest()
