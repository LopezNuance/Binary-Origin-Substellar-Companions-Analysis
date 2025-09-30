from md2unicode_math import convert_math, process


def test_convert_math_replaces_greek_and_superscript():
    assert convert_math("\\alpha + x^2") == "α + x⁲"


def test_convert_math_handles_accents():
    assert convert_math("\\hat{x}") == "x̂"


def test_process_with_keep_dollar_retains_delimiters():
    text = "Energy $$E = mc^2$$ is famous."
    result = process(text, keep_dollar=True)
    assert "$$E = mc⁲$$" in result


def test_process_without_keep_dollar_strips_delimiters():
    text = "Mass $m$"
    assert process(text) == "Mass m"

