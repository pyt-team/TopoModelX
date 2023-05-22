.. _contributing:

==================
Contributing Guide
==================

This is an exhaustive guide to ease the contribution
process for both novice and experienced contributors.

`TopoModelX <https://github.com/pyt-team/TopoModelX>`_ is a
community effort, and everyone is welcome to contribute.

.. _run tests

Run the tests
--------------

TopoModelX tests can be run using `pytest <https://docs.pytest.org/>`_.
To run tests with `pytest`, first install the required packages:

    .. code-block:: bash

      $ pip install -e .[dev]


Then run all tests using:

    .. code-block:: bash

      $ pytest test


Optionally, run a particular test file using:

    .. code-block:: bash

      $ pytest test/algorithms/<test_filename.py>


Testing
========

Test Driven Development
-------------------------

High-quality `unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_
is a corner-stone of `TopoModelX` development process.

The tests consist of classes appropriately named, located in the `test``
subdirectory, that check the validity of the algorithms and the
different options of the code.


TDD with pytest
-----------------

TopoModelX uses the `pytest` Python tool for testing different functions and features.
Install the test requirements using:

    .. code-block:: bash

      $ pip install -e .[dev]

By convention all test functions should be located in files with file names
that start with `test_`. For example a unit test that exercises the Python
addition functionality can be written as follows:

    .. code-block:: bash

      # test_add.py

      def add(x, y):
         return x + y

      def test_capital_case():
         assert add(4, 5) == 9

Use an `assert` statement to check that the function under test returns
the correct output. Then run the test using:

    .. code-block:: bash

      $ pytest test_add.py


Workflow of a contribution
===========================

The best way to start contributing is by finding a part of the project that is more familiar to you.

Alternatively, if everything is new to you and you would like to contribute while learning, look at some of the existing GitHub Issues.


.. _new-contributors:


Making changes
---------------

The preferred way to contribute to topomodelx is to fork the `upstream
repository <https://github.com/pyt-team/TopoModelX/>`__ and submit a "pull request" (PR).

Follow these steps before submitting a PR:

#. Synchronize your main branch with the upstream main branch:

    .. code-block:: bash

        $ git checkout main
        $ git pull upstream main

#. | Create a feature branch to hold your development changes:

    .. code-block:: bash

        $ git checkout -b <branch-name>

#. Make changes.

#. When you're done editing, add changed files using ``git add`` and then ``git commit``:

    .. code-block:: bash

       $ git add <modified_files>
       $ git commit -m "Add my feature"

   to record your changes. Your commit message should respect the `good
   commit messages guidelines <https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project>`_. (`How to Write a Git Commit Message <https://cbea.ms/git-commit/>`_ also provides good advice.)

   .. note::
      Before commit, make sure you have run the `black <https://github.com/psf/black>`_ and
      `flake8 <https://github.com/PyCQA/flake8>`_ tools for proper code formatting.

   Then push the changes to your fork of `TopoNextX` with:

    .. code-block:: bash

         $ git push origin <branch-name>

   Use the `-u` flag if the branch does not exist yet remotely.

#. Follow `these <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   instructions to create a pull request from your fork.

#. Repeat 3. and 4. following the reviewers requests.


It is often helpful to keep your local feature branch synchronized with the
latest changes of the main topomodelx repository. Bring remote changes locally:

    .. code-block:: bash

      $ git checkout main
      $ git pull upstream main

And then merge them into your branch:

    .. code-block:: bash

      $ git checkout <branch-name>
      $ git merge main


Pull Request Checklist
----------------------

In order to ease the reviewing process, we recommend that your contribution
complies with the following rules. The **bolded** ones are especially important:

#. **Give your pull request a helpful title.** This summarises what your
   contribution does. This title will often become the commit message once
   merged so it should summarise your contribution for posterity. In some
   cases `Fix <ISSUE TITLE>` is enough. `Fix #<ISSUE NUMBER>` is never a
   good title.

#. **Submit your code with associated unit tests**. High-quality
   `unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_
   is a corner-stone of the topomodelx development process.

#. **Make sure your code passes all unit tests**. First,
   run the tests related to your changes.

#. **Make sure that your PR follows Python international style guidelines**,
   `PEP8 <https://www.python.org/dev/peps/pep-0008>`_. The `flake8` package
   automatically checks for style violations when you
   submit your PR.

   To prevent adding commits which fail to adhere to the PEP8 guidelines, we
   include a `pre-commit <https://pre-commit.com/>`_ config, which immediately
   invokes flake8 on all files staged for commit when running `git commit`. To
   enable the hook, simply run `pre-commit install` after installing
   `pre-commit` either manually via `pip` or as part of the development requirements.

   Please avoid reformatting parts of the file that your pull request doesn't
   change, as it distracts during code reviews.

#. **Make sure that your PR follows topomodelx coding style and API** (see :ref:`coding-guidelines`). Ensuring style consistency throughout
   topomodelx allows using tools to automatically parse the codebase,
   for example searching all instances where a given function is used,
   or use automatic find-and-replace during code's refactorizations. It
   also speeds up the code review and acceptance of PR, as the maintainers
   do not spend time getting used to new conventions and coding preferences.

#. **Make sure your code is properly documented**, and **make
   sure the documentation renders properly**. To build the documentation, please
   see our :ref:`contribute_documentation` guidelines. The plugin
   flake8-docstrings automatically checks that your the documentation follows
   our guidelines when you submit a PR.

#. Often pull requests resolve one or more other issues (or pull requests).
   If merging your pull request means that some other issues/PRs should
   be closed, you should `use keywords to create link to them
   <https://github.com/blog/1506-closing-issues-via-pull-requests/>`_
   (e.g., ``fixes #1234``; multiple issues/PRs are allowed as long as each
   one is preceded by a keyword). Upon merging, those issues/PRs will
   automatically be closed by GitHub. If your pull request is simply
   related to some other issues/PRs, create a link to them without using
   the keywords (e.g., ``see also #1234``).

#. **Each PR needs to be accepted by a core developer** before being merged.


.. _contribute_documentation:

Documentation
=============

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc. reStructuredText
documents live in the source code repository under the ``docs/`` directory.

Building the Documentation
--------------------------

Building the documentation requires installing specific requirements::

   pip install -e .[doc]


Writing Docstrings
-------------------

Intro to Docstrings
~~~~~~~~~~~~~~~~~~~


A docstring is a well-formatted description of your function/class/module which includes
its purpose, usage, and other information.

There are different markdown languages/formats used for docstrings in Python. The most common
three are reStructuredText, numpy, and google docstring styles. For toponet, we are
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

N.B. Within Notes, you can
	- include LaTex code
	- cite references in text using ids placed in References


.. _docstring-examples:

Docstring Examples
~~~~~~~~~~~~~~~~~~

Here's a generic docstring template::

   def my_method(self, my_param_1, my_param_2="vector"):
      """Write a one-line summary for the method.

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

      Convenience method; equivalent to calling fit(X) followed by
      predict(X).

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