import inspect
import importlib
import re
import sys
import os

def get_class_info(module_name, class_name):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls

def format_docstring(docstring):
    if not docstring:
        return "No description available.\n\n"
    
    lines = docstring.split('\n')
    formatted = []
    in_args = False
    in_returns = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('Args:'):
            formatted.append("\n**Arguments:**\n")
            in_args = True
            in_returns = False
        elif line.startswith('Returns:'):
            formatted.append("\n**Returns:**\n")
            in_returns = True
            in_args = False
        elif in_args and ':' in line:
            param, desc = line.split(':', 1)
            formatted.append(f"- `{param.strip()}`: {desc.strip()}")
        elif in_returns and not line:
            in_returns = False
            formatted.append("\n")
        else:
            if in_args or in_returns:
                formatted.append(f"  {line}")
            else:
                formatted.append(line)
    
    return '\n'.join(formatted).strip() + '\n\n'

def pydoc_to_markdown(module_name, class_name):
    cls = get_class_info(module_name, class_name)
    
    # Class name and docstring
    md = f"## {class_name}\n\n"
    if cls.__doc__:
        md += format_docstring(cls.__doc__)
    
    # Constructor
    if cls.__init__ is not object.__init__:  # Check if __init__ is not the default
        signature = str(inspect.signature(cls.__init__))
        md += f"### Constructor\n\n#### `__init__{signature}`\n\n"
        if cls.__init__.__doc__:
            md += format_docstring(cls.__init__.__doc__)

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
                md += format_docstring(method.__doc__)

    # Get data descriptors (properties)
    properties = inspect.getmembers(cls, lambda o: isinstance(o, property))
    
    if properties:
        md += "### Properties\n\n"
        for name, prop in properties:
            if name in ['__dict__', '__weakref__']:  # Skip __dict__ and __weakref__
                continue
            
            md += f"#### `{name}`\n\n"
            
            if prop.__doc__:
                md += format_docstring(prop.__doc__)
    return md

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <module_name> <class_name>")
        sys.exit(1)

    module_name = sys.argv[1]
    class_name = sys.argv[2]

    # Add the current directory to sys.path to allow importing from the current directory
    sys.path.insert(0, os.getcwd())

    markdown_doc = pydoc_to_markdown(module_name, class_name)
    print(markdown_doc)


if __name__ == "__main__":
    main()
