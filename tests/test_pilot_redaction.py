"""Ensure pilot diagnostics do not print provider keys."""

import io
import unittest
from unittest.mock import patch

from research.decision_model.run_pilot import RedactedStderr
from research.baseline.utils import RedactedStream


class PilotRedactionTest(unittest.TestCase):
    def test_provider_error_url_is_redacted(self):
        output = io.StringIO()
        stream = RedactedStderr(output, ["dummy-secret-key"])
        stream.write("https://example.test/models/gemma?key=dummy-secret-key\n")
        self.assertEqual(
            output.getvalue(), "https://example.test/models/gemma?key=[REDACTED]\n"
        )

    def test_shared_runner_stderr_redacts_keys(self):
        output = io.StringIO()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy-secret-key"}):
            stream = RedactedStream(output)
            stream.write("https://example.test?key=dummy-secret-key\n")
        self.assertEqual(output.getvalue(), "https://example.test?key=[REDACTED]\n")
