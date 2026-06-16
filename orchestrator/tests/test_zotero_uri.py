"""Regression test: the Zotero URI regex must accept every real URI form.

The original pattern only matched the one-segment web-profile URL
(zotero.org/<user>/items/KEY) and silently rejected the standard
zotero.org/users/<id>/items/KEY form — breaking both the Zotero poller and
manually-linked papers.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.ingestion.pdf_fetcher import _ZOTERO_ATTACH_RE, _ZOTERO_PARENT_RE


def test_parent_uri_all_forms():
    forms = [
        "http://zotero.org/users/12345/items/ABCD1234",      # API/library (poller output)
        "https://www.zotero.org/users/12345/items/ABCD1234",  # "Copy Zotero URI" web form
        "https://www.zotero.org/someuser/items/ABCD1234",     # web profile (one segment)
        "zotero://select/library/items/ABCD1234",             # desktop select URI
        "https://www.zotero.org/users/12345/items/ABCD1234/attachment/WXYZ5678",
    ]
    for uri in forms:
        m = _ZOTERO_PARENT_RE.search(uri)
        assert m and m.group(1) == "ABCD1234", uri


def test_parent_uri_rejects_non_zotero():
    assert _ZOTERO_PARENT_RE.search("https://example.com/foo/items/ABCD1234") is None


def test_attachment_uri():
    m = _ZOTERO_ATTACH_RE.search(
        "https://www.zotero.org/users/12345/items/ABCD1234/attachment/WXYZ5678"
    )
    assert m and m.group(1) == "WXYZ5678"
