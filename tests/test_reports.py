import unittest

from ff_energy.latex_writer.report import Report
from ff_energy.latex_writer.figure import Figure

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def ex_fig(self) -> Figure:
        p = "/home/boittier/Documents/phd/ff_energy/latex_reports/test/energy_hist.pdf"
        f = Figure(p, "This is a test caption", "testlabel")
        return f

    def test_report_save(self):
        r = Report("test_report")
        r.set_title("Test Report")
        r.set_short_title("Test Report")
        r.set_abstract("This is a test report")
        r.add_section("Test Section")
        r.add_section(self.ex_fig())
        r.add_section("Test Section 2")
        r.write_document()
        r.compile()

