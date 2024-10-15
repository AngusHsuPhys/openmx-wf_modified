import os

def count_folders(directory):
    # Get a list of all items in the directory
    items = os.listdir(directory)
    
    # Filter the list to count only directories
    folder_count = sum(os.path.isdir(os.path.join(directory, item)) for item in items)
    
    return folder_count

if __name__ == "__main__":
    directory_path = "./mp_structures"  # Change this to your target directory if needed
    folder_count = count_folders(directory_path)
    print(f"Number of folders in '{directory_path}': {folder_count}")
