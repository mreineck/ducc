import ducc0

# Enforced by sphinx
needs_sphinx = '3.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build']
source_suffix = '.rst'
master_doc = 'index'

#######################################
# Project info
#######################################
project = u'ducc'
copyright = u'TODO'
author = 'Martin Reinecke'
# The full version, including alpha/beta/rc tags.
release = ducc0.__version__
# The short X.Y version.
version = release.split(".")[0]

#######################################
# HTML
#######################################
html_static_path = ['_static']
html_theme = 'pydata_sphinx_theme'
html_logo = '_static/ducc.jpg'

#######################################
# Napoleon
#######################################
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

#######################################
# Autosummary
#######################################
autosummary_generate = True
