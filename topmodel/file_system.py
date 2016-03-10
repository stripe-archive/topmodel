# File abstraction to allow both S3 and local to be used

import StringIO
import math
import os
import subprocess
import time

# for s3 from python
from boto.s3.connection import S3Connection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
                 security_token=None,
                 subdirectory=''):
        conn = S3Connection(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            security_token=security_token)
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

        # Take the data (a giant string of JSON), and turn it into a file
        # stream we can read from
        f = StringIO.StringIO()
        f.write(data)
        f.seek(0)

        # Count the number of chunks we're going to have to upload
        size = len(data)
        chunk_size = 1024 * 1024 * 50
        chunks = int(math.ceil(size/float(chunk_size)))

        # Start the multipart upload
        upload = self.bucket.initiate_multipart_upload(key)

        try:
            # Upload chunk by chunk, using the StringIO object instantiated above
            # we can reference the offset we want to upload.
            for i in range(chunks):
                upload.upload_part_from_file(f, part_num=i+1)
        except:
            upload.cancel_upload()
            raise
        else:
            upload.complete_upload()

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
