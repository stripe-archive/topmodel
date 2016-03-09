# File abstraction to allow both S3 and local to be used

import os
import subprocess
import time

# for s3 from python
from boto.s3 import connect_to_region

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGION = 'us-west-2'


class FileSystem(object):

    def read_file(self, path):
        raise NotImplemented

    def write_file(self, path, data):
        raise NotImplemented

    def list(self, path):
        raise NotImplemented

    def list_name_modified(self, path):
        raise NotImplemented

    def remove(self, path):
        raise NotImplemented


class S3FileSystem(FileSystem):

    def __init__(self,
                 bucket_name,
                 aws_access_key_id,
                 aws_secret_access_key,
                 region,
                 subdirectory=''):
        if region is None:
            region = DEFAULT_REGION

        conn = connect_to_region(
                region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key)
        self.bucket = conn.get_bucket(bucket_name)
        self.subdirectory = subdirectory
        if subdirectory and not subdirectory.endswith('/'):
            self.subdirectory += '/'

    def read_file(self, path):
        key = self.bucket.get_key(self.subdirectory + path)
        if key is None:
            return None
        return key.read()

    def write_file(self, path, data):
        key = self.bucket.get_key(self.subdirectory + path)
        if key is None:
            key = self.bucket.new_key(self.subdirectory + path)
        key.set_contents_from_string(data)

    def list(self, path=''):
        subdirlen = len(self.subdirectory)
        return [key.name[subdirlen:] for key in self.bucket.list(self.subdirectory + path)]

    def list_name_modified(self, path=''):
        model_names_and_modified = {}
        subdirlen = len(self.subdirectory)
        for key in self.bucket.list(self.subdirectory + path):
            model_names_and_modified[key.name[subdirlen:]] = key.last_modified
        return model_names_and_modified

    def remove(self, path):
        keys = self.bucket.get_all_keys(prefix=self.subdirectory + path)
        self.bucket.delete_keys(keys)


class LocalFileSystem(FileSystem):

    def __init__(self, basedir=None):
        if basedir is None:
            basedir = PROJECT_ROOT
        self.basedir = basedir

    def read_file(self, path):
        try:
            with open(self.abspath(path), 'r') as f:
                return f.read()
        except IOError:
            return None

    def write_file(self, path, data):
        # Make the intermediate directories if they don't exist
        path_dir = os.path.dirname(self.abspath(path))
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        with open(self.abspath(path), 'w') as f:
            return f.write(data)

    def list(self, path=''):
        walker = os.walk(os.path.join(self.basedir, path))
        return [
            os.path.join(dirpath, filename)[len(self.basedir + '/'):]
            for dirpath, _, filenames in walker
            for filename in filenames
        ]

    def list_name_modified(self, path=''):
        model_names_and_modified = {}
        filenames = self.list()
        for name in filenames:
            model_names_and_modified[name] = time.ctime(os.path.getctime(name))
        return model_names_and_modified

    def remove(self, path):
        subprocess.check_call(["rm", "-r", self.abspath(path)])

    def abspath(self, path):
        return os.path.join(self.basedir, path)
