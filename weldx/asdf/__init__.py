"""This submodule contains ASDF related weldx extensions and schemas."""
# isort:skip_file
from weldx.asdf import tags  # implement tags before the asdf extensions
from weldx.asdf import constants, util
from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension

# class imports to weldx.asdf namespace
from weldx.asdf.tags.weldx.core.file import ExternalFile
