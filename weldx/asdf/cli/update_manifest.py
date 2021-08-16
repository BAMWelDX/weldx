from pathlib import Path

import yaml

from weldx.asdf.types import _legacy_tag_from_new_tag
from weldx.asdf.util import get_converter_for_uri


def update_manifest(
    search_dir: str = "../schemas", out: str = "../manifests/weldx-1.0.0.yaml"
):
    """Create manifest file from existing schemas."""
    # read existing manifest
    manifest = yaml.load(
        Path(out).read_text(),
        Loader=yaml.SafeLoader,
    )

    # keep only ASDF schema mappings
    manifest["tags"] = [
        mapping
        for mapping in manifest["tags"]
        if mapping["schema_uri"].startswith("http://stsci.edu/schemas")
    ]

    schemas = Path(search_dir).rglob("*.yaml")

    for schema in schemas:
        content = yaml.load(
            schema.read_text(),
            Loader=yaml.SafeLoader,
        )
        if "id" in content:  # should be schema file
            uri: str = content["id"]
            tag = uri.replace("/schemas/", "/tags/")
            if get_converter_for_uri(tag):  # check if converter is implemented
                manifest["tags"].append(dict(tag_uri=tag, schema_uri=uri))
                manifest["tags"].append(
                    dict(tag_uri=_legacy_tag_from_new_tag(tag), schema_uri=uri)
                )  # legacy_tag
            else:
                print(f"No converter for URI: {schema}")

    with open(Path(out), "w") as outfile:
        outfile.write("%YAML 1.1\n---\n")
        yaml.dump(
            manifest,
            outfile,
            default_flow_style=False,
            sort_keys=False,
        )
        outfile.write("...\n")


if __name__ == "__main__":
    from weldx.asdf.constants import MANIFEST_PATH, SCHEMA_PATH

    update_manifest(SCHEMA_PATH, MANIFEST_PATH / "weldx-1.0.0.yaml")
