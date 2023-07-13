project = 'thebeat'
copyright = '2023, Jelle van der Werff, Andrea Ravignani, & Yannick Jadoul'
author = 'Jelle van der Werff, Andrea Ravignani, & Yannick Jadoul'

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
    'nbsphinx',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autosectionlabel'
    ]

# bibtex
bibtex_bibfiles = ['refs.bib']
bibtext_reference_style = 'author_year'

templates_path = ['_templates']
exclude_patterns = ['_build', '_templates']


intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None),
                       'parselmouth': ('https://parselmouth.readthedocs.io/en/stable/', None),
                       'sounddevice': ('https://python-sounddevice.readthedocs.io/en/latest/', None),
                       'pandas': ('https://pandas.pydata.org/docs/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/', None),
                       'abjad': ('https://abjad.github.io/', None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "extra_navbar": "<p>Package developed by the <a href='https://www.mpi.nl/department/comparative-bioacoustics/20'>Comparative Bioacoustics</a> group.</p>",
    "show_toc_level": 3,
    "collapse_navigation": True,
    'logo_only': True
}
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_logo = "thebeat_logo.png"
html_favicon = 'favicon.ico'


# default role for easy lookup using `whatever`
# not using it now; bit confusing I think.
#default_role = 'py:obj'


# Napoleon settings
napoleon_include_init_with_doc = True

napoleon_preprocess_types = True
napoleon_use_rtype = True

# autosummary
autosummary_generate = True

# autodoc
autodoc_default_flags = ['members']

# Type hints settings
typehints_defaults = 'comma'
