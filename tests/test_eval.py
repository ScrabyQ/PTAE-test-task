import copy
import json
import unittest

from eval import check_ledger, score
from returns_agent.data import DEFAULT_DATA, load_runtime


class EvalTest(unittest.TestCase):
    def test_reference_null_and_line_order_and_fields_outside_score(self):
        ref = {"request_id": "X", "decision": "refund", "order_id": "O-1", "refund": {
            "lines": [{"sku": "A", "qty": 1, "amount": 20}, {"sku": "B", "qty": 2, "amount": 8}],
            "total": 28, "method": "SBP",
        }}
        actual = copy.deepcopy(ref)
        actual.update(proposed_refund_total=None, message_to_customer="оформлено")
        actual["refund"]["method"] = "another method"
        actual["refund"]["lines"].reverse()
        self.assertEqual(score([ref], [actual]), (1, []))
        actual["refund"]["lines"][0]["qty"] = True
        actual["refund"]["total"] = 27
        actual.pop("proposed_refund_total")
        matched, diff = score([ref], [actual])
        self.assertEqual(matched, 0)
        self.assertEqual({d[1] for d in diff}, {"refund.lines", "refund.total", "proposed_refund_total"})
        self.assertEqual(score([ref], [actual, actual])[1], [("X", "results.count", 1, 2)])
        self.assertEqual(score([ref], [])[1], [("X", "results.count", 1, 0)])

    def test_method_checked_independently_and_export_must_agree(self):
        data = load_runtime(DEFAULT_DATA)
        # Use a published golden only in this eval test, never to populate a runtime directory.
        example = next(x for x in json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))["golden"]
                       if x["decision"] == "refund")
        row = {**example, "proposed_refund_total": None, "message_to_customer": "Оформлен возврат.",
               "note_for_staff": None}
        row["refund"] = {**row["refund"], "method": "wrong"}
        scoped = data.model_copy(update={"requests": [r for r in data.requests if r.request_id == row["request_id"]]})
        problems = check_ledger(scoped, [row], [r.model_dump() for r in data.refunds])
        self.assertTrue(any("refund.method" in x for x in problems))
        self.assertTrue(any("операци" in x for x in problems))
