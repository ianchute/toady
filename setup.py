from distutils.core import setup
setup(
  name = 'toady',
  packages = ['toady'], # this must be the same as the name above
  version = '1.10',
  description = 'Easily visualize high-dimensional data in 2d space',
  author = 'Ian Chu Te',
  author_email = 'ianchute@hotmail.com',
  url = 'https://github.com/ianchute/toady', # use the URL to the github repo
  download_url = 'https://github.com/ianchute/toady/archive/1.10.tar.gz', # I'll explain this in a second
  keywords = ['visualization', 'embedding', 'data science', 'data'], # arbitrary keywords
  classifiers = [],
  install_requires=[
      'numpy',
      'pandas',
      'matplotlib',
      'mpld3',
      'sklearn',
  ],
)