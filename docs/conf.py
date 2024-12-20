"""Sphinx configuration file."""

import os
import shutil

# -- Project information -----------------------------------------------------

project = "TopoModelX"
copyright = "2022-2023, PyT-Team, Inc."
author = "PyT-Team Authors"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "toponetx": ("https://pyt-team.github.io/toponetx/", None),
}

# Configure nbsphinx for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True


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
        if len(b) != 0:
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

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_baseurl = "https://pyt-team.github.io/topomodelx/"

html_context = {
    "github_user": "pyt-team",
    "github_repo": "TopoModelX",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pyt-team/TopoModelX",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": True,
}

html_favicon = "_static/favicon-48.png"

html_show_sourcelink = False

# Exclude copy button from appearing over notebook cell numbers by using :not()
# The default copybutton selector is `div.highlight pre`
# https://github.com/executablebooks/sphinx-copybutton/blob/master/sphinx_copybutton/__init__.py#L82
copybutton_selector = ":not(.prompt) > div.highlight pre"

# -- Options for EPUB output -------------------------------------------------

epub_exclude_files = ["search.html"]
