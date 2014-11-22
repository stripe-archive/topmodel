import shutil
import tempfile
import unittest
from os.path import join

from topmodel.model_data import ModelData
from topmodel.file_system import LocalFileSystem


class ModelDataTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir_path = tempfile.mkdtemp()
        self.file_system = LocalFileSystem(self.tmpdir_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir_path)

    def test_get_notes(self):
        model_data = ModelData(self.file_system, 'my_model_name')
        shutil.copytree('./data/test/my_model_name',
                        join(self.tmpdir_path, 'my_model_name'))
        assert model_data.get_notes() == "Test model notes\n"

    def test_get_missing_notes(self):
        model_data = ModelData(self.file_system, 'my_other_model_name')
        shutil.copytree('./data/test/my_other_model_name',
                        join(self.tmpdir_path, 'my_other_model_name'))
        assert model_data.get_notes() is None

    def test_set_notes(self):
        shutil.copytree('./data/test/my_other_model_name',
                        join(self.tmpdir_path, 'my_other_model_name'))
        note = "My note"
        model_data = ModelData(self.file_system, 'my_other_model_name')
        model_data.set_notes(note)
        print model_data.get_notes()
        assert model_data.get_notes() == note
