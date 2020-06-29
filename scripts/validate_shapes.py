"""Tests to validate shapes."""

from weldx.asdf.validators import _custom_shape_validator as val
from weldx.asdf.validators import _another_validator as an_val

# should be working
assert val([3], [3]) is True

assert val([2, 4, 5], [2, 4, 5]) is True

assert val([1, 2, 3], ["..."]) is True

assert val([1, 2], [1, 2, "..."]) is True

assert val([1, 2, 3], [1, 2, None]) is True

assert val([1, 2, 3], [1, 2, "(~)"]) is True
assert val([1, 2, 3], [1, 2, "(:)"]) is True

assert val([1, 2], [1, 2, "(:)"]) is True
assert val([1, 2], [1, 2, "(~)"]) is True

assert val([1], [1, "..."]) is True

assert val([1, 2], [1, 2, "(3)"]) is True

assert val([1, 2, 3], [1, 2, 3]) is True

assert val([1, 2, 3], [1, "1:3", 3]) is True
assert val([1, 2, 3], [1, "1~3", 3]) is True

assert val([1, 2, 3], [1, "1:", 3]) is True
assert val([1, 2, 3], [1, "1~", 3]) is True

assert val([1, 2, 3], [1, ":3", 3]) is True
assert val([1, 2, 3], [1, "~3", 3]) is True

assert val([1, 2, 3], [1, "..."]) is True

assert val([1, 2, 3], ["...", 3]) is True
assert val([1, 2, 3], ["(1)", 2, 3]) is True
assert val([1, 2, 3], ["(1)", "(2)", 3]) is True


# values are wrong
assert val([2, 2, 3], [1, "..."]) is False

assert val([1], [1, 2]) is False

assert val([1, 2], [1]) is False

assert val([1, 2], [3, 2]) is False

assert val([1], [1, ":"]) is False
assert val([1], [1, "~"]) is False
assert val([1], [1, None]) is False

assert val([1, 2, 3], [1, 2, "(4)"]) is False

assert val([1, 2], [1, "4:8"]) is False
assert val([1, 2], [1, "4~8"]) is False

# expected values are wrong
try:
    # val([1, 2], [1, ":", "(...)"])
    val([1, 2], [1, "~", "(...)"])
except ValueError as err:
    print(err)
    print()

try:
    val([1, 2], [1, "(2)", 3])
except ValueError as err:
    print(err)
    print()

try:
    val([1, 2], [1, "...", 2])
except ValueError as err:
    print(err)
    print()

try:
    # seems to be unintuitive
    val([1, 2], ["(1)", "..."])
except ValueError as err:
    print(err)
    print()

try:
    val([1, 2], [1, "...2"])
except ValueError as err:
    print(err)
    print()

try:
    # "x:y" => (x <= y)
    # val([1, 2], [1, "4:1"])
    val([1, 2], [1, "4~1"])
except ValueError as err:
    print(err)
    print()

# ----------------------------------------------------------------------------------
# test _another_validator
dict_test = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 3, 3]}
dict_expected = {"a": [1, "n", 3], "b": [1, "n", "m"], "c": ["m", 3, 3]}
assert an_val(dict_test, dict_expected)

dict_test = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 3, 3]}
dict_expected = {"a": [1, "n", "..."], "b": ["...", "n", "m"], "c": ["m", 3, "(3)"]}
assert an_val(dict_test, dict_expected)

dict_test = {"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 3], "c": [3, 3, 3]}
dict_expected = {"a": [1, "n", "..."], "b": [1, "n", "m"], "c": ["m", 3, 3]}
assert an_val(dict_test, dict_expected)

dict_test = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 3, 3]}
dict_expected = {"a": [1, "n", 3], "b": [1, "n", "(m)"], "c": ["m", 3, "(~)"]}
assert an_val(dict_test, dict_expected)

dict_test = {"a": [1, 2, 3], "b": [1, 2], "c": [3, 3]}
dict_expected = {"a": [1, "n", 3], "b": [1, "n", "(m)"], "c": ["m", 3, "(:)"]}
assert an_val(dict_test, dict_expected)

dict_test = {"a": [1, 2, 3], "b": [1, 2], "c": [3, 3]}
dict_expected = {"a": [1, "n", None], "b": [1, "n", "(m)"], "c": ["m", 3, "(~)"]}
assert an_val(dict_test, dict_expected)

try:
    dict_test = {"a": [1, 2, 3]}
    dict_expected = {"a": [1, "~", "(...)"]}
    an_val(dict_test, dict_expected)
except ValueError as err:
    print(err)
    print()

try:
    dict_test = {"a": [1, 2]}
    dict_expected = {"a": [1, "(2)", 3]}
    an_val(dict_test, dict_expected)
except ValueError as err:
    print(err)
    print()

try:
    dict_test = {"a": [1, 2]}
    dict_expected = {"a": [1, "...", 2]}
    an_val(dict_test, dict_expected)
except ValueError as err:
    print(err)
    print()

try:
    # seems to be unintuitive
    dict_test = {"a": [1, 2]}
    dict_expected = {"a": ["(1)", "..."]}
    an_val(dict_test, dict_expected)
except ValueError as err:
    print(err)
    print()

try:
    dict_test = {"a": [1, 2]}
    dict_expected = {"a": [1, "...2"]}
    an_val(dict_test, dict_expected)
except ValueError as err:
    print(err)
    print()

try:
    # "x~y" => (x <= y)
    dict_test = {"a": [1, 2]}
    dict_expected = {"a": [1, "4~1"]}
    an_val(dict_test, dict_expected)
except ValueError as err:
    print(err)
    print()

print("EOF success")
