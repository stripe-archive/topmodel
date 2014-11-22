from topmodel_server import app
import unittest


class TopModelRouteTest(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.local = True
        self.app = app.test_client()
        self.app.local = True

    def test_compare(self):
        resp = self.app.get(
            '/compare?model[]=data/test/my_model_name/&model[]=data/test/my_other_model_name/')
        html = resp.data
        assert 'stroke-width' in html

    def test_basic_model(self):
        resp = self.app.get('/model/data/test/my_model_name/')
        html = resp.data
        assert 'stroke-width' in html
        # Make sure the notes are displayed on the page
        assert 'Test model notes' in html

    def test_integer_targets(self):
        """Test that it doesn't crash if the targets are '0' instead of 'False'"""
        resp = self.app.get('/model/data/test/integer_targets/')
        html = resp.data
        assert 'stroke-width' in html
