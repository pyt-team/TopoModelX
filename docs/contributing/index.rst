.. _contributing:

============
Contributing
============

This guide aims to ease the contribution
process for both novice and experienced contributors.

`TopoModelX <https://github.com/pyt-team/TopoModelX>`_ is a
community effort, and everyone is welcome to contribute.

Making changes
--------------

The preferred way to contribute to topomodelx is to fork the `upstream
repository <https://github.com/pyt-team/TopoModelX/>`__ and submit a "pull request" (PR).

Follow these steps before submitting a PR:

#. Synchronize your main branch with the upstream main branch:

    .. code-block:: bash

        $ git checkout main
        $ git pull upstream main

#. Create a feature branch to hold your development changes:

    .. code-block:: bash

        $ git checkout -b <branch-name>

#. Make changes. Make sure that you provide appropriate unit-tests and documentation to your code. See next sections of this contributing guide for details.

#. When you're done editing, add changed files using ``git add`` and then ``git commit``:

    .. code-block:: bash

       $ git add <modified_files>
       $ git commit -m "Add my feature"

   to record your changes. Then push the changes to your fork of `TopoNextX` with:

    .. code-block:: bash

         $ git push origin <branch-name>

#. Follow `these <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   instructions to create a pull request from your fork.

#. Repeat 3. and 4. following the reviewers requests.

Write tests
-----------

The tests consist of classes appropriately named, located in the `test` folder, that check the validity of the code.

Test functions should be located in files whose filenames start with `test_`. For example:

    .. code-block:: bash

      # test_add.py

      def add(x, y):
         return x + y

      def test_capital_case():
         assert add(4, 5) == 9

Use an `assert` statement to check that the function under test returns the correct output. 

Run tests
~~~~~~~~~

Install `pytest` which is the software tools used to run tests:

    .. code-block:: bash
    
      $ pip install -e .[dev]

Then run the test using:

    .. code-block:: bash

      $ pytest test_add.py
      
Verify that the code you have added does not break `TopoModelX` by running all the tests.

    .. code-block:: bash
    
      $ pytest test/

Write Documentation
-------------------

Building the documentation requires installing specific requirements.

    .. code-block:: bash
    
      $ pip install -e .[doc]

Intro to Docstrings
~~~~~~~~~~~~~~~~~~~

A docstring is a well-formatted description of your function/class/module which includes
its purpose, usage, and other information.

There are different markdown languages/formats used for docstrings in Python. The most common
three are reStructuredText, numpy, and google docstring styles. For topomodelx, we are
using the numpy docstring standard.
When writing up your docstrings, please review the `NumPy docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
to understand the role and syntax of each section. Following this syntax is important not only for readability,
it is also required for automated parsing for inclusion into our generated API Reference.

You can look at these for any object by printing out the ``__doc__`` attribute.
Try this out with the np.array class and the np.mean function to see good examples::

>>> import numpy as np
>>> print(np.mean.__doc__)

The Anatomy of a Docstring
~~~~~~~~~~~~~~~~~~~~~~~~~~

These are some of the most common elements for functions (and ones we’d like you to add where appropriate):

#. Summary - a one-line (here <79 char) description of the object

   a. Begins immediately after the first """ with a capital letter, ends with a period

   b. If describing a function, use a verb with the imperative mood (e.g. **Compute** vs Computes)

   c. Use a verb which is as specific as possible, but default to Compute when uncertain (as opposed to Calculate or Evaluate, for example)

#. Description - a more informative multi-line description of the function

   a. Separated from the summary line by a blank line

   b. Begins with a capital letter and ends with period

#. Parameters - a formatted list of arguments with type information and description

   a. On the first line, state the parameter name, type, and shape when appropriate. The parameter name should be separated from the rest of the line by a ``:`` (with a space on either side). If a parameter is optional, write ``Optional, default: default_value.`` as a separate line in the description.
   b. On the next line, indent and write a summary of the parameter beginning with a capital letter and ending with a period.

   c. See :ref:`docstring-examples`.

#. Returns (esp. for functions) - a formatted list of returned objects type information and description

   a. The syntax here is the same as in the parameters section above.

   b. See :ref:`docstring-examples`.

If documenting a class, you would also want to include an Attributes section.
There are many other optional sections you can include which are very helpful.
For example: Raises, See Also, Notes, Examples, References, etc.

N.B. Within Notes, you can:

- include LaTex code
- cite references in text using ids placed in References

Docstring Examples
~~~~~~~~~~~~~~~~~~

Here's a generic docstring template::

   def my_method(self, my_param_1, my_param_2="vector"):
      r"""Write a one-line summary for the method.

      Write a description of the method, including "big O"
      (:math:`O\left(g\left(n\right)\right)`) complexities.

      Parameters
      ----------
      my_param_1 : array-like, shape=[..., dim]
         Write a short description of parameter my_param_1.
      my_param_2 : str, {"vector", "matrix"}
         Write a short description of parameter my_param_2.
         Optional, default: "vector".

      Returns
      -------
      my_result : array-like, shape=[..., dim, dim]
         Write a short description of the result returned by the method.

      Notes
      -----
      If relevant, provide equations with (:math:)
      describing computations performed in the method.

      Example
      -------
      Provide code snippets showing how the method is used.
      You can link to scripts of the examples/ directory.

      Reference
      ---------
      If relevant, provide a reference with associated pdf or
      wikipedia page.
      """

And here's a filled-in example from the Scikit-Learn project, modified to our syntax::

   def fit_predict(self, X, y=None, sample_weight=None):
       """Compute cluster centers and predict cluster index for each sample.

       Convenience method; equivalent to calling fit(X) followed by predict(X).

       Parameters
       ----------
       X : {array-like, sparse_matrix} of shape=[..., n_features]
          New data to transform.
       y : Ignored
          Not used, present here for API consistency by convention.
       sample_weight : array-like, shape [...,], optional
          The weights for each observation in X. If None, all observations
          are assigned equal weight (default: None).

       Returns
       -------
       labels : array, shape=[...,]
          Index of the cluster each sample belongs to.
       """
       return self.fit(X, sample_weight=sample_weight).labels_

In general, have the following in mind:

   #. Use built-in Python types. (``bool`` instead of ``boolean``)

   #. Use ``[`` for defining shapes: ``array-like, shape=[..., dim]``

   #. If a shape can vary, use a list-like notation:
      ``array-like, shape=[dimension[:axis], n, dimension[axis:]]``

   #. For strings with multiple options, use brackets:
      ``input: str, {"log", "squared", "multinomial"}``

   #. 1D or 2D data can be a subset of
      ``{array-like, ndarray, sparse matrix, dataframe}``. Note that
      ``array-like`` can also be a ``list``, while ``ndarray`` is explicitly
      only a ``numpy.ndarray``.

   #. Add "See Also" in docstrings for related classes/functions.
      "See Also" in docstrings should be one line per reference,
      with a colon and an explanation.

For Class and Module Examples see the `scikit-learn _weight_boosting.py module
<https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/ensemble/_weight_boosting.py#L285>`_.
The class AdaBoost has a great example using the elements we’ve discussed here.
Of course, these examples are rather verbose, but they’re good for
understanding the components.

When editing reStructuredText (``.rst``) files, try to keep line length under
80 characters (exceptions include links and tables).
