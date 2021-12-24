[metadata]
name = text-based-clusterer
version = 0.0.3
author = Benedict Taguinod
author_email = benedict.a.taguinod@gmail.com
description = A package that clusters python objects based on a chosen string attribute
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/pypa/text-based-clusterer
project_urls =
    Bug Tracker = https://github.com/pypa/text-based-clusterer/issues
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.6

[options.packages.find]
where = src