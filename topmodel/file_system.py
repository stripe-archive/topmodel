# File abstraction to allow both S3 and local to be used

import os
import subprocess

# for s3 from python
import boto

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FileSystem(object):

    def read_file(self, path):
        raise NotImplemented

    def write_file(self, path, data):
        raise NotImplemented

    def list(self):
        raise NotImplemented

    def remove(self, path):
        raise NotImplemented


class S3FileSystem(object):

    def __init__(self, bucket_name, aws_access_key_id, aws_secret_access_key):
        conn = boto.connect_s3(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key)
        self.bucket = conn.get_bucket(bucket_name)

    def read_file(self, path):
        key = self.bucket.get_key(path)
        if key is None:
            return None
        return key.read()

    def write_file(self, path, data):
        key = self.bucket.get_key(path)
        if key is None:
            key = self.bucket.new_key(path)
        key.set_contents_from_string(data)

    def list(self):
        return [key.name for key in self.bucket.list()]

    def remove(self, path):
        keys = self.bucket.get_all_keys(prefix=path)
        self.bucket.delete_keys(keys)


class LocalFileSystem(object):

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

    def list(self):
        return [
            os.path.join(dirpath, filename)[len(self.basedir + '/'):]
            for dirpath, _, filenames in os.walk(self.basedir)
            for filename in filenames
        ]

    def remove(self, path):
        subprocess.check_call(["rm", "-r", self.abspath(path)])

    def abspath(self, path):
        return os.path.join(self.basedir, path)
