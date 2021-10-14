from pathlib import Path

import yaml

from weldx.asdf.util import get_converter_for_tag


def update_manifest(
    search_dir: str = "../../weldx/schemas",
    out: str = "../../weldx/manifests/weldx-0.1.0.yaml",
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
    update_manifest()
