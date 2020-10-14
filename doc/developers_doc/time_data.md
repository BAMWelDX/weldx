# Time Data

## Why is time special?

An important property of time that distinguishes it from most other dimensions we use is that it is constantly growing.
In consequence, we usually expect time dependent data to be ordered. 
In addition, time has a unique unit system with varying conversion factors: Seconds, Hours, Days, Years, etc.
While other dimensions also have unit systems with varying conversion factors that are not powers of 10 (1 mile = 1760 
yards, 1 yard = 3 feet), it is not so common to mix them when specifying a quantity, especially when using SI units.
Instead of writing `q = 5km + 169m + 33mm` one would usually express it using a single unit, for example: 
`q = 5,169033km`.
On the other hand, we often specify a certain point in time in form of a timestamp like `2016-12-06 17:36:48` instead of
using a fractional number of years.

There is certainly a lot more to write about what makes time special, but the details are not really relevant for this
document. 

## Design principles

### Supported data types

The `weldx` package interfaces support `pandas.TimedeltaIndex`, `pandas.DatetimeIndex`, `pandas.Timestamp` and 
`pint.Quantity` as time data types.
To keep things as simple as possible we are not planning on extending this list in the near future.
It is up to the user to take care of other types by casting them to one of the supported data types.

### Which type should be used internally?

The preferred data type for internal usage is `pandas.TimedeltaIndex`.
In combination with a `pandas.Timestamp` as reference time it can store the same information as a 
`pandas.DatetimeIndex` but it can also be used without a reference time.
This makes it more flexible to use than the `pandas.DatetimeIndex` that has a "baked in" reference time that must always
be specified.
Often we are only interested in the temporal relations towards a specific event (like the start of an experiment) and 
not when exactly this event occurred.
In this case the combination of `pandas.TimedeltaIndex` and the optional `pandas.Timestamp` enables us to omit 
irrelevant data.

Even though a `pint.Quantity` stores the same information as a `pandas.TimedeltaIndex`, it doesn't enforce a temporal 
order.
Therefore `pandas.TimedeltaIndex` is better suited as internal data format.

Apart from the officially supported data types, `NumPy` time data types are used internally for some specific tasks like
serialization.
The main reason for this is that the `pandas` types are based on them. 


### How should time be handled in public interfaces?

Each function that accepts time as parameter should be able to deal with the supported data types that were previously
specified.
This usually means that the corresponding function actually needs two parameters, `time` and `reference_time`.
The reference time is optional when using `pandas.TimedeltaIndex` or a `pint.Quantity` as `time` and is ignored if 
`time` is a `pandas.DatetimeIndex`.

> TODO: discuss return type

The favored return type of public functions and class properties is a `pint.Quantity` since it is the most important
data type for the user.
Ideally, he is never required to deal with any other type. 

## Common operations and their solutions

This section covers some frequently required operations that already have a default solution that should be utilized.

### Type casting 

An obvious task that often occurs is casting between the different time data types. 
There are already two functions implemented that can be found in the `weldx.utility` package.

The first one is `to_pandas_time_index`.
Depending on the input variable type, it either returns an `pandas.DatetimeIndex` or a `pandas.TimedeltaIndex`.
It accepts all of the supported data types as arguments but some others like the `NumPy` time data types are also
accepted.
Check the API documentation for further information.
The second function is called `pandas_time_delta_to_quantity` and casts, as the name suggests, a `pandas.TimedeltaIndex`
into a `pint.Quantity`.

In case you want to split a `pandas.DatetimeIndex` into a time delta and a reference time, you can simply subtract the
`pandas.Timestamp` of your choice from the `pandas.DatetimeIndex` to get a corresponding `pandas.TimedeltaIndex`.