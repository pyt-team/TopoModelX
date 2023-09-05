"""Sphinx configuration file."""

import os
import shutil

import topomodelx

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

language = "en"

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = None

html_theme = "pydata_sphinx_theme"
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


def copy_thumbnails():
    """Copy the thumbnail files in the _build
    directory to enable thumbnails in the gallery"""
    src_directory = "./_thumbnails"
    des_directory = "./_build/_thumbnails"

    des_directory_walked = os.walk(src_directory)
    all_thumbnails = []

    for a, b, c in des_directory_walked:
        if len(c) == 0:
            all_directories = b
            continue
        elif len(b) != 0:
            raise NotImplementedError(
                "Not yet implemented for the case with more than one nested directory."
            )

        for file in c:
            full_filename = a + "/" + file
            all_thumbnails.append(full_filename)

    os.mkdir("./_build")
    os.mkdir(des_directory)

    for directory in all_directories:
        os.mkdir(des_directory + "/" + directory)

    for thumbnail in all_thumbnails:
        shutil.copyfile(thumbnail, "./_build/" + thumbnail[2:])


copy_thumbnails()

nbsphinx_thumbnails = {
    "notebooks/cell/can_train_bis": "_thumbnails/cell/can_train_bis.png",
    "notebooks/cell/can_train": "_thumbnails/cell/can_train.png",
    "notebooks/cell/ccxn_train": "_thumbnails/cell/ccxn_train.png",
    "notebooks/cell/cwn_train": "_thumbnails/cell/cwn_train.png",
    "notebooks/hypergraph/allset_train": "_thumbnails/hypergraph/allset_train.png",
    "notebooks/hypergraph/allset_transformer_train": "_thumbnails/hypergraph/allset_transformer_train.png",
    "notebooks/hypergraph/dhgcn_train": "_thumbnails/hypergraph/dhgcn_train.png",
    "notebooks/hypergraph/hmpnn_train": "_thumbnails/hypergraph/hmpnn_train.png",
    "notebooks/hypergraph/hnhn_train_bis": "_thumbnails/hypergraph/hnhn_train_bis.png",
    "notebooks/hypergraph/hnhn_train": "_thumbnails/hypergraph/hnhn_train.png",
    "notebooks/hypergraph/hypergat_train": "_thumbnails/hypergraph/hypergat_train.png",
    "notebooks/hypergraph/hypersage_train": "_thumbnails/hypergraph/hypersage_train.png",
    "notebooks/hypergraph/template_train": "_thumbnails/hypergraph/template_train.png",
    "notebooks/hypergraph/unigcn_train": "_thumbnails/hypergraph/unigcn_train.png",
    "notebooks/hypergraph/unigcnii_train": "_thumbnails/hypergraph/unigcnii_train.png",
    "notebooks/hypergraph/unigin_train": "_thumbnails/hypergraph/unigin_train.png",
    "notebooks/hypergraph/unisage_train": "_thumbnails/hypergraph/unisage_train.png",
    "notebooks/simplicial/dist2cycle_train": "_thumbnails/simplicial/dist2cycle_train.png",
    "notebooks/simplicial/hsn_train": "_thumbnails/simplicial/hsn_train.png",
    "notebooks/simplicial/san_train": "_thumbnails/simplicial/san_train.png",
    "notebooks/simplicial/sca_cmps_train": "_thumbnails/simplicial/sca_cmps_train.png",
    "notebooks/simplicial/sccn_train": "_thumbnails/simplicial/sccn_train.png",
    "notebooks/simplicial/sccnn_train": "_thumbnails/simplicial/sccnn_train.png",
    "notebooks/simplicial/scconv_train": "_thumbnails/simplicial/scconv_train.png",
    "notebooks/simplicial/scn2_train": "_thumbnails/simplicial/scn2_train.png",
    "notebooks/simplicial/scnn_train": "_thumbnails/simplicial/scnn_train.png",
    "notebooks/simplicial/scone_train_bis": "_thumbnails/simplicial/scone_train_bis.png",
    "notebooks/simplicial/scone_train": "_thumbnails/simplicial/scone_train.png",
}
