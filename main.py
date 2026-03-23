# main.py

# IMPORTS
import view
import sys

# MAIN
major = sys.version_info[0]
minor = sys.version_info[1]
version = f"{major}.{minor}"

if version == "3.13":
    v = view.View()
    v.run()
else:
    print("Please use python version 3.13.")