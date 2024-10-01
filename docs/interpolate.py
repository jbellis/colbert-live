import sys
import os
import re

def interpolate_file(file_path, script_dir):
    with open(file_path, 'r') as f:
        content = f.read()

    def replace_file_content(match):
        relative_path = match.group(1)
        full_path = os.path.normpath(os.path.join(script_dir, relative_path))
        try:
            with open(full_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: File not found: {full_path}", file=sys.stderr)
            return f"{{File not found: {relative_path}}}"

    # Use a regex to find and replace {filename} patterns
    interpolated_content = re.sub(r'\{([^}]+)\}', replace_file_content, content)
    return interpolated_content

def main():
    if len(sys.argv) != 3:
        print("Usage: python interpolate.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change working directory to the script's directory
    os.chdir(script_dir)

    try:
        interpolated_content = interpolate_file(input_file, script_dir)
        
        with open(output_file, 'w') as f:
            f.write(interpolated_content)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
