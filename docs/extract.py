import ast
import sys


def extract_elements(filename, targets):
    with open(filename, 'r') as file:
        source = file.read()

    tree = ast.parse(source)

    extracted_elements = []

    for node in ast.iter_child_nodes(tree):
        node_type = type(node).__name__
        node_name = getattr(node, 'name', None)

        if node_type in targets or (node_name and node_name in targets):
            extracted_elements.append(ast.unparse(node))

    return extracted_elements


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <filename> <target1> <target2> ...")
        sys.exit(1)

    filename = sys.argv[1]
    targets = sys.argv[2:]

    elements = extract_elements(filename, targets)
    print('\n'.join(elements))


if __name__ == "__main__":
    main()
