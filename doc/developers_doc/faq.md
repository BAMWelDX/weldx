# FAQ

## CI pipelines / Github actions

**The "check notebooks" action fails. How do I remove all outputs from an ipython notebook?** 

As described in [this Stack Overflow answer](https://stackoverflow.com/a/47774393/6700329), simply run

~~~
jupyter nbconvert --clear-output --inplace my_notebook.ipynb
~~~