"""Classes and functions to configure the WelDX package."""

from pathlib import Path
from typing import Dict, List, Union

import asdf
import pkg_resources
import yaml
from asdf.config import ResourceMappingProxy
from asdf.versioning import AsdfVersion, split_tag_version

from weldx.types import types_path_like


class QualityStandard:
    """Stores information about a quality standard."""

    def __init__(self, resource_root_dir: types_path_like):
        """Create a ``QualityStandard`` instance.

        Parameters
        ----------
        resource_root_dir :
            The path to the resource root directory of the standard

        """
        from fs.osfs import OSFS

        self._name = None
        self._max_version = None
        self._versions = {}

        if isinstance(resource_root_dir, Path):
            resource_root_dir = resource_root_dir.as_posix()

        if isinstance(resource_root_dir, str):
            self._filesystem = OSFS(resource_root_dir)
        else:
            self._filesystem = resource_root_dir

        manifest_dir = self._filesystem.opendir("manifests")
        manifest_files = [
            file.name
            for file in self._filesystem.filterdir("manifests", ["*.yml", "*.yaml"])
        ]

        for filename in manifest_files:
            # stem of pyfilesystem cuts after first .
            qs_name, version = split_tag_version(filename[: filename.rindex(".")])

            if self._name is None:
                self._name = qs_name
                self._max_version = version
            else:
                if qs_name != self._name:
                    raise ValueError("Inconsistent naming of manifest files")
                if self._max_version < version:
                    self._max_version = version

            with manifest_dir.open(filename, "r") as stream:
                content = yaml.load(stream, Loader=yaml.SafeLoader)
                self._versions[version] = {
                    "manifest_file_mapping": {content["id"]: filename},
                    "schema_file_mapping": {
                        mapping["uri"]: (f"{mapping['file']}.yaml")
                        for mapping in content["tags"]
                    },
                }

    def _map_file_content(
        self, file_mapping: Dict, directory: str, version: AsdfVersion
    ) -> ResourceMappingProxy:
        """Get a mapping between an URI and a file content.

        Parameters
        ----------
        file_mapping : Dict
            A dictionary containing the mapping between URI and the file path
        directory:
            Directory that contains the files. This is either 'schemas' or 'mappings'
        version : AsdfVersion
            The version of the standard.

        Returns
        -------
        ResourceMappingProxy :
            Mapping between an URI and a file content

        """
        content_mapping = {
            uri: self._filesystem.open(f"{directory}/{filename}").read()
            for uri, filename in file_mapping.items()
        }

        return ResourceMappingProxy(
            content_mapping, package_name=self._name, package_version=version
        )

    @property
    def name(self) -> str:
        """Get the quality standards name."""
        return self._name

    def get_mappings(self, version: Union[AsdfVersion, str] = None):
        """Get the manifest and schema mapping for the specified version.

        Parameters
        ----------
        version : Union[AsdfVersion, str]
            Requested standard version. If `None` is provided, the latest will be used.

        Returns
        -------
        ResourceMappingProxy :
            Manifest mapping
        ResourceMappingProxy :
            Schema mapping

        """
        if version is None:
            version = self._max_version
        elif not isinstance(version, AsdfVersion):
            version = AsdfVersion(version)

        file_mappings = self._versions[version]
        manifest_mapping = self._map_file_content(
            file_mappings["manifest_file_mapping"], "manifests", version
        )
        schema_mapping = self._map_file_content(
            file_mappings["schema_file_mapping"], "schemas", version
        )

        return manifest_mapping, schema_mapping


class Config:
    """Manages the global configuration."""

    _standards = {}

    @staticmethod
    def add_quality_standard(standard: QualityStandard):
        """Register a quality standard.

        Parameters
        ----------
        standard :
            Quality standard that should be added

        """
        Config._standards[standard.name] = standard

    @staticmethod
    def enable_quality_standard(name: str, version: Union[AsdfVersion, str] = None):
        """Enable a quality standard.

        All corresponding schemas will be used for validation during serialization and
        deserialization of a weldx file.

        Parameters
        ----------
        name :
            Name of the quality standard
        version :
            Requested standard version. If `None` is provided, the latest will be used.

        """
        standard = Config._standards[name]
        manifest_mapping, schema_mapping = standard.get_mappings(version)
        asdf_config = asdf.get_config()
        asdf_config.add_resource_mapping(manifest_mapping)
        asdf_config.add_resource_mapping(schema_mapping)

    @staticmethod
    def load_installed_standards():
        """Load all standards that are installed to the active virtual environment."""
        for entry_point in pkg_resources.iter_entry_points("weldx.standard"):
            standards = entry_point.load()()
            if not isinstance(standards, List):
                standards = [standards]
            for standard in standards:
                if not isinstance(standard, QualityStandard):
                    raise TypeError("Invalid quality standard.")
                Config.add_quality_standard(standard)


def add_quality_standard(standard: QualityStandard):
    """Register a quality standard.

    Parameters
    ----------
    standard :
        Quality standard that should be added

    """
    Config.add_quality_standard(standard)


def enable_quality_standard(name: str, version: Union[AsdfVersion, str] = None):
    """Enable a quality standard.

    All corresponding schemas will be used for validation during serialization and
    deserialization of a weldx file.

    Parameters
    ----------
    name :
        Name of the quality standard
    version :
        Requested standard version. If `None` is provided, the latest will be used.

    """
    Config.enable_quality_standard(name, version)
