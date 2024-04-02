import os
import bitsandbytes
import sys

# Get the directory of the bitsandbytes module
module_dir = os.path.dirname(bitsandbytes.__file__)

# Print the location of the bitsandbytes module
print(f"bitsandbytes module location: {module_dir}")

# List the contents of the module directory
print(f"Contents of {module_dir}:")
for item in os.listdir(module_dir):
    print(item)

# Print the Python Path (sys.path)
print("Python Path (sys.path):")
for path in sys.path:
    print(f"  {path}")