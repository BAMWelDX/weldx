"""Tests to validate shapes."""

from weldx.asdf.validators import _custom_shape_validator as val

# should be working
assert val("3", "3") == True

assert val("2,4,5", "2,4,5") == True

assert val("2, 4, 5", "2,4,5") == True

assert val("1,2,3", "...") == True

assert val("1,2", "1,2,...") == True

assert val("1,2,3", "1,2,:") == True

assert val("1,2,3", "1,2,(:)") == True

assert val("1,2", "1,2,(:)") == True

assert val("1", "1,...") == True

assert val("1,2", "1,2,(3)") == True

assert val("1,2,3", "1,2,3") == True

assert val("1,2,3", "1,1:3,3") == True

assert val("1,2,3", "1,1:,3") == True

assert val("1,2,3", "1,:3,3") == True

assert val("1,2,3", "1,...") == True


# values are wrong
assert val("2,2,3", "1,...") == False

assert val("1", "1,2") == False

assert val("1,2", "1") == False

assert val("1,2", "3,2") == False

assert val("1", "1,:") == False

assert val("1,2,3", "1,2,(4)") == False

assert val("1,2", "1,4:8") == False

# expected values are wrong
try:
    val("1,2", "1,:,(...)")
except ValueError as err:
    print(err)
    print()

try:
    val("1,2", "1,(2),3")
except ValueError as err:
    print(err)
    print()

try:
    val("1,2", "1,...,2")
except ValueError as err:
    print(err)
    print()

try:
    # seems to be unintuitive
    val("1,2", "(1),...")
except ValueError as err:
    print(err)
    print()

try:
    val("1,2", "1,...2")
except ValueError as err:
    print(err)
    print()

try:
    # "x:y" => (x <= y)
    val("1,2", "1,4:1")
except ValueError as err:
    print(err)
    print()

print("EOF success")
