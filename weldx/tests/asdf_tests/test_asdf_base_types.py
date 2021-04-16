"""Tests asdf implementations of python base types."""
import uuid

import pytest
from asdf import ValidationError

from weldx.asdf.util import write_read_buffer


# --------------------------------------------------------------------------------------
# uuid
# --------------------------------------------------------------------------------------
def test_uuid():
    """Test uuid serialization and version 4 pattern."""
    write_read_buffer({"id": uuid.uuid4()})

    with pytest.raises(ValidationError):
        write_read_buffer({"id": uuid.uuid1()})
