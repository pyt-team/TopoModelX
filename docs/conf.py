"""Sphinx configuration file."""

project = "TopoModelX"
copyright = "2022-2023, PyT-Team, Inc."
author = "PyT-Team Authors"

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.load_style",
]

# Configure napoleon for numpy docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = False

# Configure nbsphinx for notebooks execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"

# To get a prompt similar to the Classic Notebook, use
nbsphinx_input_prompt = " In [%s]:"
nbsphinx_output_prompt = " Out [%s]:"

nbsphinx_allow_errors = True

templates_path = ["_templates"]

source_suffix = [".rst"]

master_doc = "index"

language = None

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = None

html_baseurl = "pyt-team.github.io"
htmlhelp_basename = "pyt-teamdoc"
html_last_updated_fmt = "%c"

latex_elements = {}


latex_documents = [
    (
        master_doc,
        "topomodelx.tex",
        "TopoModelX Documentation",
        "PyT-Team",
        "manual",
    ),
]

man_pages = [(master_doc, "topomodelx", "TopoModelX Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "topomodelx",
        "TopoModelX Documentation",
        author,
        "topomodelx",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]
