from pathlib import Path

import yaml


def create_manifest(
    search_dir: str = "../schemas", out: str = "../manifests/weldx_manifest-1.0.0.yaml"
):
    """Create manifest file from existing schemas."""
    manifest = dict(
        id="asdf://weldx.bam.de/weldx/manifests/weldx-1.0.0",
        extension_uri="asdf://weldx.bam.de/weldx/extensions/weldx-1.0.0",
        title="weldx extension manifest for tag mapping",
        description=None,
        tags=[],
    )

    schemas = Path(search_dir).rglob("*.yaml")

    for schema in schemas:
        content = yaml.load(
            schema.read_text(),
            Loader=yaml.SafeLoader,
        )
        if "id" in content and "tag" in content:
            manifest["tags"].append(
                dict(tag_uri=content["tag"], schema_uri=content["id"])
            )
        else:
            print(f"skipping {schema}")

    with open(Path(out), "w") as outfile:
        outfile.write("%YAML 1.1\n---\n")
        yaml.dump(manifest, outfile, default_flow_style=False, sort_keys=False)
        outfile.write("...\n")


if __name__ == "__main__":
    from weldx.asdf.constants import SCHEMA_PATH

    create_manifest(SCHEMA_PATH)
