import argparse
import os

from .file_access import SAVED_MODEL_FOLDER

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename a saved model')
    parser.add_argument('-o', '--old-name', required=True, help='The old name of the model.')
    parser.add_argument('-n', '--new-name', required=True, help='The new name of the model.')
    args = parser.parse_args()

    for file_name in os.listdir(SAVED_MODEL_FOLDER):
        if file_name.startswith(args.old_name + '.'):
            os.rename(os.path.join(SAVED_MODEL_FOLDER, file_name), os.path.join(SAVED_MODEL_FOLDER, file_name.replace(args.old_name, args.new_name)))