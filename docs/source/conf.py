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
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []


intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# for ipynb notebooks
nbsphinx_input_prompt = 'In [%s]:'


# Napoleon settings
napoleon_include_init_with_doc = True

napoleon_preprocess_types = True
napoleon_use_rtype = True
typehints_defaults = 'comma'
