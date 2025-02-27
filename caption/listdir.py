import os

# Path to the directory you want to list subdirectories from
dir_path = "/home/dj191/research/dataset/classification/val"

# Check if the directory exists
if os.path.exists(dir_path) and os.path.isdir(dir_path):
    # List all entries in the directory
    entries = os.listdir(dir_path)
    
    # Filter out only directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(dir_path, entry))]
    directories.sort()
    
    # Print the directory names
    if directories:
        print("Directories in '{}':".format(dir_path))
        for directory in directories:
            print(directory)
    else:
        print("No directories found in '{}'.".format(dir_path))
else:
    print("The path '{}' does not exist or is not a directory.".format(dir_path))