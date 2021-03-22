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
html_theme_options = {
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/ducc0",
            "icon": "fas fa-box",
        }
    ],
    "gitlab_url": "https://gitlab.mpcdf.mpg.de/mtr/ducc",
}

#######################################
# Napoleon
#######################################
napoleon_google_docstring = False
napoleon_numpy_docstring = True

#######################################
# Autosummary
#######################################
autosummary_generate = True
