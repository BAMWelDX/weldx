"""Tests to validate shapes."""

from weldx.asdf.validators import _custom_shape_validator as an_val

# test _custom_shape_validator
dict_test = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 3, 3]}
dict_expected = {"a": [1, "n", 3], "b": [1, "n", "m"], "c": ["m", 3, 3]}
assert an_val(dict_test, dict_expected)

dict_test = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 3, 3]}
dict_expected = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 3, 3]}
assert an_val(dict_test, dict_expected) == {}

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

dict_test = {
    "a": {"a1": [1, 2, 3], "a2": {"a21": [3, 2, 1], "a22": [2, 2, 2]}},
    "b": {"b1": [1, 2, 3], "b2": [1, 1, 1]},
    "c": [3, 3, 3],
}
dict_expected = {
    "a": {"a1": [1, "n", 3], "a2": {"a21": ["m", 2, 1], "a22": ["n", 2, "n"]}},
    "b": {"b1": [1, "~", "m"], "b2": [1, 1, 1]},
    "c": ["m", 3, 3],
}
assert an_val(dict_test, dict_expected)

dict_test = {  # here is a 4 -------------v
    "a": {"a1": [1, 2, 3], "a2": {"a21": [4, 2, 1], "a22": [2, 2, 2]}},
    "b": {"b1": [1, 2, 3], "b2": [1, 1, 1]},
    "c": [3, 3, 3],
}
dict_expected = {
    "a": {"a1": [1, "n", 3], "a2": {"a21": ["m", 2, 1], "a22": ["n", 2, "n"]}},
    "b": {"b1": [1, 2, "m"], "b2": [1, 1, 1]},
    "c": ["m", 3, 3],
}
assert an_val(dict_test, dict_expected) is False

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
