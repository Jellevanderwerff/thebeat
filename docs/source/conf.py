project = 'Combio package'
copyright = '2022, Jelle van der Werff & Yannick Jadoul'
author = 'Jelle van der Werff & Yannick Jadoul'

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex'
    ]



templates_path = ['_templates']
exclude_patterns = []


intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None), 
                       'parselmouth': ('https://parselmouth.readthedocs.io/en/stable/', None), 
                       'sounddevice': ('https://python-sounddevice.readthedocs.io/', None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "extra_navbar": "<p>Package developed by the <a href='https://www.mpi.nl/department/comparative-bioacoustics/20'>Comparative Bioacoustics</a> group.</p>",
    "show_toc_level": 2,
    "collapse_navigation": True
}
html_static_path = ['_static']



# Napoleon settings
napoleon_include_init_with_doc = True

napoleon_preprocess_types = True
napoleon_use_rtype = True
typehints_defaults = 'comma'

# bibtex
bibtex_bibfiles = ['refs.bib']
bibtext_reference_style = 'author_year'
