import yaml

# Load the YAML file
with open('bigdata.yaml', 'r') as file:
    env_data = yaml.safe_load(file)

# Extract dependencies
dependencies = env_data['dependencies']

# Function to simplify package format
def simplify_package(package):
    if '=' in package:
        parts = package.split('=')
        return f"{parts[0]}={parts[1]}"
    return package

# Simplify the packages
simplified_dependencies = [simplify_package(dep) for dep in dependencies if isinstance(dep, str)]

# Print the simplified packages
for dep in simplified_dependencies:
    print(dep)

