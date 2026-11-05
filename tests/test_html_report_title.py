import importlib.util
from pathlib import Path
import tempfile
import unittest
from html import escape


MODULE_PATH = Path(__file__).resolve().parents[1] / "quant_reporter" / "html_builder.py"
SPEC = importlib.util.spec_from_file_location("html_builder", MODULE_PATH)
html_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(html_builder)


class ReportTitleTests(unittest.TestCase):
    def render(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "report.html"
            html_builder.generate_html_report(filename=filename, **kwargs)
            return filename.read_text(encoding="utf-8")

    def test_title_is_text_in_document_title_and_heading(self):
        for title in ['Research & Development <2026>', '<script>alert("title")</script>', 'Café "Growth"']:
            with self.subTest(title=title):
                report = self.render(sections=[], title=title)
                self.assertIn(f"<title>{escape(title)}</title>", report)
                self.assertIn(f"<h1>{escape(title)}</h1>", report)

    def test_default_title_is_preserved(self):
        report = self.render(sections=[])
        self.assertIn("<title>Quantitative Report</title>", report)
        self.assertIn("<h1>Quantitative Report</h1>", report)

    def test_explicit_table_html_is_preserved(self):
        table = '<table><tr><td>A &amp; B</td></tr></table>'
        sections = [{"title": "Summary", "main_content": [{"type": "table_html", "data": table}]}]
        report = self.render(sections=sections, title="A & B")
        self.assertIn(table, report)


if __name__ == "__main__":
    unittest.main()
