"""Classes and functions to configure the WelDX package."""

import os
from pathlib import Path
from typing import Dict, List, Union

import asdf
import pkg_resources
import yaml
from asdf import generic_io
from asdf.config import ResourceMappingProxy
from asdf.versioning import AsdfVersion, split_tag_version


class QualityStandard:
    """Stores information about a quality standard."""

    def __init__(self, resource_root_dir: Path):
        """Create a `QualityStandard` instance

        Parameters
        ----------
        resource_root_dir : Path
            The path to the resource root directory of the standard

        """
        self._name = None
        self._max_version = None
        self._versions = {}

        manifest_file_paths = self._get_manifest_file_paths(resource_root_dir)

        for path in manifest_file_paths:
            filename, version = split_tag_version(path.stem)
            if self._name is None:
                self._name = filename
                self._max_version = version
            else:
                if filename != self._name:
                    raise ValueError("Inconsistent naming of manifest files")
                if self._max_version < version:
                    self._max_version = version

            with open(path, "r") as stream:
                content = yaml.load(stream, Loader=yaml.SafeLoader)
                mappings = self._get_schema_mappings(content, resource_root_dir)
                self._versions[version] = {
                    "manifest_file_mapping": {content["id"]: path},
                    "schema_file_mapping": mappings,
                }

    @staticmethod
    def _get_manifest_file_paths(resource_root_dir) -> List:
        """Get a list of all manifest files path's.

        Parameters
        ----------
        resource_root_dir : Path
            The path to the resource root directory of the standard

        Returns
        -------
        List :
           A list of all manifest files path's

        """
        manifest_dir = resource_root_dir / "manifests"
        return [
            Path(os.path.join(manifest_dir, file))
            for file in os.listdir(manifest_dir)
            if file.endswith((".yml", ".yaml"))
        ]

    @staticmethod
    def _get_schema_mappings(manifest_dict: Dict, resource_root_dir: Path) -> Dict:
        """Get all schema file mappings from the manifest file.

        Parameters
        ----------
        manifest_dict : Dict
            Content of the manifest file as dictionary
        resource_root_dir : Path
            The path to the resource root directory of the standard

        Returns
        -------
        Dict :
            Schema file mappings

        """
        return {
            mapping["uri"]: (resource_root_dir / "schemas" / f"{mapping['file']}.yaml")
            for mapping in manifest_dict["tags"]
        }

    def _map_file_content(
        self, file_mapping: Dict, version: AsdfVersion
    ) -> ResourceMappingProxy:
        """ Get a mapping between an URI and a file content.

        Parameters
        ----------
        file_mapping : Dict
            A dictionary containing the mapping between URI and the file path
        version : AsdfVersion
            The version of the standard.

        Returns
        -------
        ResourceMappingProxy :
            Mapping between an URI and a file content

        """
        content_mapping = {
            uri: generic_io.get_file(file_path).read().decode("utf-8")
            for uri, file_path in file_mapping.items()
        }

        return ResourceMappingProxy(
            content_mapping, package_name=self._name, package_version=version
        )

    @property
    def name(self) -> str:
        """Get the quality standards name.

        Returns
        -------
        str :
            Name of the quality standard
        """
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
            Scheme mapping

        """
        if version is None:
            version = self._max_version
        elif not isinstance(version, AsdfVersion):
            version = AsdfVersion(version)

        file_mappings = self._versions[version]
        manifest_mapping = self._map_file_content(
            file_mappings["manifest_file_mapping"], version
        )
        schema_mapping = self._map_file_content(
            file_mappings["schema_file_mapping"], version
        )

        return manifest_mapping, schema_mapping


class Config:
    """Manages the global configuration."""

    _standards = {}

    @staticmethod
    def _add_quality_standard(standard: QualityStandard):
        """Register a quality standard.

        Parameters
        ----------
        standard : QualityStandard
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
        name : str
            Name of the quality standard
        version : Union[AsdfVersion, str]
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
                Config._add_quality_standard(standard)
