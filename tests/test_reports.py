import unittest

from ff_energy.latex_writer.report import Report

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_report_save(self):
        r = Report("test_report")
        r.set_title("Test Report")
        r.set_short_title("Test Report")
        r.set_abstract("This is a test report")
        r.add_section("Test Section")
        r.write_document()
        r.compile()

