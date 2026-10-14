"""Tests for the browser read-aloud text preparation.

Only ``clean_for_speech`` and the HTML assembly are testable here; the actual
synthesis runs in the browser via the Web Speech API and is verified manually.
"""

from src.ui import speaker


def test_strips_markdown_syntax():
    out = speaker.clean_for_speech("## Title\n\nThe **bold** and *italic* text.")
    assert "**" not in out and "*" not in out and "#" not in out
    assert "bold" in out and "italic" in out


def test_removes_citation_markers():
    out = speaker.clean_for_speech("Subsidy is 100% [1]. Also see [2, 3] for more.")
    assert "[1]" not in out and "[2, 3]" not in out
    assert "Subsidy is 100%" in out


def test_removes_urls_but_keeps_link_text():
    out = speaker.clean_for_speech(
        "Apply at https://tn.gov.in/x or via [the portal](https://example.com)."
    )
    assert "https://" not in out
    assert "the portal" in out


def test_no_orphan_punctuation_after_stripping():
    # Removing "[1]" must not leave "farmers ." which reads as an odd pause.
    out = speaker.clean_for_speech("Helps small farmers [1]. Yield rose [2].")
    assert " ." not in out
    assert ".." not in out


def test_truncates_on_sentence_boundary():
    out = speaker.clean_for_speech("This is a sentence. " * 100, max_chars=200)
    assert len(out) <= 201
    assert out.endswith(".")


def test_empty_input_is_safe():
    assert speaker.clean_for_speech("") == ""
    assert speaker.speak_html("   ") == "<div></div>"


def test_speak_html_fills_all_placeholders():
    html = speaker.speak_html("Drip irrigation subsidy is fifty percent.")
    assert "<script>" in html
    for token in ("__CFG__", "__AUTO__", "__BTN__", "__NONCE__"):
        assert token not in html


def test_quotes_and_script_tags_cannot_break_out():
    # Answer text is model output, so it must be JSON-encoded into the script.
    html = speaker.speak_html('He said "hi" </script> and \\ backslash')
    assert "</script> and" not in html
    assert html.count("</script>") == 1


def test_autoplay_flag_is_respected():
    assert "if (true)" in speaker.speak_html("x", autoplay=True)
    assert "if (false)" in speaker.speak_html("x", autoplay=False)
