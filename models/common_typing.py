from typing import *

Either = Union
Path = NewType("Path", str)

# Unix-style path which can include wildcards (eg *). See `glob https://docs.python.org/3/library/glob.html` for more information.
UnixStylePath = NewType("Unix_Style_Path", str)

# See https://docs.python.org/3/library/string.html#formatstrings amd https://pypi.org/project/parse/
FormatString = NewType("Format_String", str)
