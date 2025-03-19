import os


def is_exists_dir(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' exists and is a directory.")
        return True
    else:
        print(f"The directory '{directory_path}' does not exist or is not a directory.")
        return False


def is_exists_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        print(f"The file '{file_path}' exists.")
        return True
    else:
        print(f"The file '{file_path}' does not exist.")
        return False


def delete_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"The file '{file_path}' has been deleted.")
    except OSError as e:
        print(f"Error deleting the file '{file_path}': {str(e)}")


def delete_files_in_dir(directory_path):  # delete all files in a directory
    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all files in the directory
        file_list = os.listdir(directory_path)

        # Iterate over the files and delete them
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                else:
                    print(f"Skipped: {file_path} (not a file)")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
    else:
        print(f"The directory '{directory_path}' does not exist or is not a directory.")
