from pathlib import Path

from cc_pipeline.clean.pdf_cleaner import pdf_to_text


FIXTURES = Path(__file__).parent / "fixtures"


def test_pdf_to_text_extracts_text():
    pdf_path = FIXTURES / "hello.pdf"
    pdf_bytes = pdf_path.read_bytes()

    text = pdf_to_text(pdf_bytes)

    assert isinstance(text, str)
    assert "Hello" in text