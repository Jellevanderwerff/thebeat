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
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'myst_parser',
    'nbsphinx'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Napoleon settings
napoleon_include_init_with_doc = True