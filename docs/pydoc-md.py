import inspect
import importlib
import re
import sys
import os

def get_class_info(module_name, class_name):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls

def pydoc_to_markdown(module_name, class_name):
    cls = get_class_info(module_name, class_name)
    
    # Class name and docstring
    md = f"## {class_name}\n\n"
    if cls.__doc__:
        md += f"{cls.__doc__.strip()}\n\n"
    
    # Constructor
    if cls.__init__ is not object.__init__:  # Check if __init__ is not the default
        signature = str(inspect.signature(cls.__init__))
        md += f"### Constructor\n\n#### `__init__{signature}`\n\n"
        if cls.__init__.__doc__:
            docstring = re.sub(r'\n\s*', '\n', cls.__init__.__doc__.strip())
            md += f"{docstring}\n\n"
        else:
            md += "No description available.\n\n"
    
    # Get methods
    methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    
    if methods:
        md += "### Methods\n\n"
        for name, method in methods:
            if name.startswith('_') or name == '__init__':  # Skip private methods and constructor
                continue
            
            # Method signature
            signature = str(inspect.signature(method))
            md += f"#### `{name}{signature}`\n\n"
            
            # Method docstring
            if method.__doc__:
                # Clean up the docstring
                docstring = re.sub(r'\n\s*', '\n', method.__doc__.strip())
                md += f"{docstring}\n\n"
            else:
                md += "No description available.\n\n"
    
    # Get data descriptors (properties)
    properties = inspect.getmembers(cls, lambda o: isinstance(o, property))
    
    if properties:
        md += "### Properties\n\n"
        for name, prop in properties:
            if name in ['__dict__', '__weakref__']:  # Skip __dict__ and __weakref__
                continue
            
            md += f"#### `{name}`\n\n"
            
            if prop.__doc__:
                docstring = re.sub(r'\n\s*', '\n', prop.__doc__.strip())
                md += f"{docstring}\n\n"
            else:
                md += "No description available.\n\n"
    
    return md

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <module_name> <class_name>")
        sys.exit(1)

    module_name = sys.argv[1]
    class_name = sys.argv[2]

    # Add the current directory to sys.path to allow importing from the current directory
    sys.path.insert(0, os.getcwd())
    
    try:
        markdown_doc = pydoc_to_markdown(module_name, class_name)
        
        # Print to stdout
        print(markdown_doc)
    
    except ImportError:
        print(f"Error: Could not import module '{module_name}'", file=sys.stderr)
        sys.exit(1)
    except AttributeError:
        print(f"Error: Class '{class_name}' not found in module '{module_name}'", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
