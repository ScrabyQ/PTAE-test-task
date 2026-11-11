import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from returns_agent.contracts import AgentError, AssessArgs, IntentItem, TOOL_ARGS, parse_tool_args, tool_schema
from returns_agent.data import DEFAULT_DATA, load_runtime


class ContractsTest(unittest.TestCase):
    def test_runtime_has_no_golden_and_no_unscoped_tool_parameters(self):
        data = load_runtime(DEFAULT_DATA)
        self.assertNotIn("golden", type(data).model_fields)
        self.assertTrue(data.requests)
        for name, dto in TOOL_ARGS.items():
            schema = tool_schema(name)["function"]["parameters"]
            self.assertIs(schema["additionalProperties"], False)
            if dto is not AssessArgs:
                self.assertEqual(schema["properties"], {})
                with self.assertRaises(ValidationError):
                    dto.model_validate({"request_id": "another-request"})

    def test_unknowns_are_explicit_and_quantities_are_strict(self):
        item = {"sku": None, "qty": None, "reason": "unclear", "evidence": "товар"}
        IntentItem.model_validate(item)
        for value in (True, 0, -1, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                IntentItem.model_validate({**item, "qty": value})
        with self.assertRaises(ValidationError):
            IntentItem.model_validate({**item, "amount": 9000})
        with self.assertRaises(ValidationError):
            IntentItem.model_validate({k: v for k, v in item.items() if k != "qty"})

    def test_wire_args_reject_duplicate_or_unknown_fields(self):
        self.assertEqual(parse_tool_args("create_refund", "{}").model_dump(), {})
        for encoded in ('{"amount":1}', '{"intent":null,"intent":null}', '[]', ''):
            with self.subTest(encoded=encoded), self.assertRaisesRegex(AgentError, "^INVALID_ARGUMENTS$"):
                parse_tool_args("assess_return", encoded)

    def test_changed_policy_cannot_run_with_old_rules(self):
        source = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))
        source["policy"][0]["text"] += " Правило изменено."
        # No private source records are needed for a version check.
        for section in ("requests", "customers", "orders", "refunds", "golden"):
            source[section] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(AgentError, "^UNSUPPORTED_POLICY$"):
                load_runtime(path)
