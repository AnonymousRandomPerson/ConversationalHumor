import argparse
import os

from send2trash import send2trash

from .file_access import SAVED_MODEL_FOLDER

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delete a saved model')
    parser.add_argument('-m', '--model-file', required=True, help='The name of the model to delete.')
    args = parser.parse_args()

    for file_name in os.listdir(SAVED_MODEL_FOLDER):
        if file_name.startswith(args.model_file + '.'):
            send2trash(os.path.join(SAVED_MODEL_FOLDER, file_name))