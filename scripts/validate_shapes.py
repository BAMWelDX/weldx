"""Tests to validate shapes."""

from weldx.asdf.validators import _custom_shape_validator as val

# should be working
val("3", "3")

val("2,4,5", "2,4,5")

val("2, 4, 5", "2,4,5")

val("1,2,3", "...")

val("1,2", "1,2,...")

val("1,2,3", "1,2,:")

val("1", "1,...")

val("1,2", "1,2,(3)")

val("1,2,3", "1,2,3")

val("1,2,3", "1,1:3,3")

val("1,2,3", "1,1:,3")

val("1,2,3", "1,:3,3")

val("1,2,3", "1,...")


# values are wrong
try:
    val("2,2,3", "1,...")
except AssertionError as err:
    print(err)
    print()

try:
    val("1", "1,2")
except IndexError as err:
    print(err)
    print()

try:
    val("1,2", "1")
except AssertionError as err:
    print(err)
    print()

try:
    val("1,2", "3,2")
except AssertionError as err:
    print(err)
    print()

try:
    val("1", "1,:")
except IndexError as err:
    print(err)
    print()

try:
    val("1,2,3", "1,2,(4)")
except AssertionError as err:
    print(err)
    print()

try:
    val("1,2", "1,4:8")
except AssertionError as err:
    print(err)
    print()

# expected values are wrong
try:
    val("1,2", "1,:,(...)")
except AssertionError as err:
    print(err)
    print()

try:
    val("1,2", "1,(2),3")
except AssertionError as err:
    print(err)
    print()

try:
    val("1,2", "1,...,2")
except AssertionError as err:
    print(err)
    print()

try:
    # seems to be unintuitive
    val("1,2", "(1),...")
except AssertionError as err:
    print(err)
    print()

try:
    val("1,2", "1,...2")
except AssertionError as err:
    print(err)
    print()

try:
    # "x:y" => (x <= y)
    val("1,2", "1,4:1")
except AssertionError as err:
    print(err)
    print()

print("EOF success")
