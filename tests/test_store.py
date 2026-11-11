"""Local fault and race tests. Each test starts with its own output directory."""

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from returns_agent.contracts import AgentError, Assessment, PrivateRequest, Refund, RuntimeData
from returns_agent.store import RefundStore, _atomic_replace


def source():
    return RuntimeData.model_validate({
        "meta": {"description": "Тест", "as_of": "2026-01-20", "timezone": "Europe/Moscow",
                 "currency": "RUB", "auto_refund_limit": 50000},
        "policy": [], "catalog": [
            {"sku": "S-A", "name": "Чайник", "category": "kitchen", "price": 800,
             "warranty_months": 12, "returnable": True},
            {"sku": "S-B", "name": "Кабель", "category": "accessories", "price": 90,
             "warranty_months": 6, "returnable": True},
        ],
        "customers": [{"customer_id": "C-A", "full_name": "Storage Canary", "email": "ledger@example.test",
                       "phone": "canary_phone", "address": "canary_address"}],
        "orders": [{"order_id": "O-A", "customer_id": "C-A", "ordered_at": "2026-01-10",
                    "delivered_at": "2026-01-15", "payment_method": "SBP", "delivery_cost": 100,
                    "lines": [{"sku": "S-A", "qty": 1, "paid_unit_price": 700},
                              {"sku": "S-B", "qty": 3, "paid_unit_price": 80}]}],
        "refunds": [{"refund_id": "RF-0001", "order_id": "O-A", "request_id": "legacy",
                     "created_at": "2026-01-17", "method": "SBP", "total": 80,
                     "lines": [{"sku": "S-B", "qty": 1, "amount": 80}]}],
        "requests": [],
    })


def request(key="new", at="2026-01-20T10:00:00+03:00"):
    return PrivateRequest(request_id=key, received_at=at, channel="email",
                          from_email="ledger@example.test", text="Возврат. private_text_canary")


def approval(sku="S-A", qty=1):
    amount = {"S-A": 700, "S-B": 80}[sku] * qty
    refund = Refund.model_validate({"method": "SBP", "total": amount,
                                    "lines": [{"sku": sku, "qty": qty, "amount": amount}]})
    return Assessment(decision="refund", order_id="O-A", reason_code="APPROVED",
                      refund=refund, policy_rules=["P1", "P4", "P6"], proposed_refund_total=None)


def declined(decision="reject"):
    return Assessment(decision=decision, order_id="O-A", policy_rules=["P7"],
                      reason_code="ALREADY_RETURNED" if decision == "reject" else "EXCESS_QUANTITY",
                      refund=None, proposed_refund_total=None)


def pay(store, req, assessment):
    return store.commit(req, assessment, create=True, recheck=lambda refunds: assessment)


def competing_writer(out, request_id, ready, decisions):
    store = RefundStore(Path(out), source())
    authorized = approval()

    def recheck(refunds):
        if any(line.sku == "S-A" for refund in refunds for line in refund.lines):
            return declined()
        return authorized

    ready.wait(timeout=20)
    try:
        result = store.commit(request(request_id), authorized, recheck=recheck, create=True)
        decisions.put(result.decision)
    except AgentError as exc:
        decisions.put(exc.code)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.data = source()
        self.store = RefundStore(self.out, self.data)

    def read(self, name):
        return json.loads((self.out / name).read_text("utf-8"))

    def assert_code(self, code, fn, *, committed=False):
        with self.assertRaises(AgentError) as error:
            fn()
        self.assertEqual(error.exception.code, code)
        self.assertEqual(error.exception.committed, committed)

    def test_legacy_and_same_tool_after_lost_response(self):
        req, a = request(), approval()
        self.assertIsNone(self.store.lookup(req))
        confirmed = pay(self.store, req, a)
        self.assertEqual(confirmed.refund, a.refund)
        self.assertEqual(self.store.commit(req, a, create=True, recheck=lambda _: self.fail("rechecked")), confirmed)
        self.assertEqual(RefundStore(self.out, self.data).lookup(req), confirmed)
        refunds = self.read("refunds.json")
        self.assertEqual(len(refunds), 2)
        self.assertEqual(refunds[0], self.data.refunds[0].model_dump())
        self.assertEqual(self.read("results.json"), [confirmed.model_dump()])
        ledger = (self.out / "state.json").read_text("utf-8")
        for sensitive in ("ledger@example.test", "private_text_canary", "canary_address", "Storage Canary"):
            self.assertNotIn(sensitive, ledger)

    def test_conflicts_preserve_state_and_exports(self):
        req = request()
        pay(self.store, req, approval())
        original = {name: (self.out / name).read_bytes() for name in ("state.json", "results.json", "refunds.json")}
        changed = req.model_copy(update={"text": "другая просьба"})
        self.assert_code("IDEMPOTENCY_CONFLICT", lambda: self.store.lookup(changed))
        self.assert_code("IDEMPOTENCY_CONFLICT", lambda: pay(self.store, req, approval("S-B")))
        self.assert_code("IDEMPOTENCY_CONFLICT", lambda: self.store.lookup(request("legacy")))
        for name, content in original.items():
            self.assertEqual((self.out / name).read_bytes(), content)

    def test_recovery_after_commit_between_exports(self):
        req, a = request(), approval()

        def interrupt(path, content):
            if path.name == "results.json":
                raise OSError("test fault")
            return _atomic_replace(path, content)

        with patch("returns_agent.store._atomic_replace", side_effect=interrupt):
            self.assert_code("EXPORT_FAILED_COMMITTED", lambda: pay(self.store, req, a), committed=True)
        self.assertEqual(len(self.read("refunds.json")), 2)
        self.assertEqual(self.read("results.json"), [])
        confirmed = self.store.commit(req, a, create=True, recheck=lambda _: self.fail("rechecked"))
        self.assertEqual(self.read("results.json"), [confirmed.model_dump()])
        self.assertEqual(len(self.store.refunds()), 2)
        (self.out / "refunds.json").write_text("stale export", encoding="utf-8")
        (self.out / "results.json").unlink()
        self.assertEqual(RefundStore(self.out, self.data).lookup(req), confirmed)
        self.assertEqual(len(self.read("refunds.json")), 2)
        self.assertEqual(self.read("results.json"), [confirmed.model_dump()])

    def test_failed_state_replace_preserves_previous_ledger(self):
        initial = (self.out / "state.json").read_bytes()
        with patch("returns_agent.store.os.replace", side_effect=OSError("test fault")):
            self.assert_code("STATE_WRITE_FAILED", lambda: pay(self.store, request(), approval()))
        self.assertEqual((self.out / "state.json").read_bytes(), initial)
        self.assertIsNone(self.store.lookup(request()))
        self.assertEqual(len(self.read("refunds.json")), 1)
        self.assertEqual(self.read("results.json"), [])
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_all_decisions_replay_in_received_order(self):
        # The earlier event has a lexically later local clock time.
        later = request("r-later", "2026-01-20T13:30:00+03:00")
        earlier = request("r-earlier", "2026-01-20T14:05:00+04:00")
        self.store.commit(later, declined("escalate"))
        self.store.commit(earlier, declined())
        self.assertEqual([r["request_id"] for r in self.read("results.json")], ["r-earlier", "r-later"])
        for req, decision in ((later, "escalate"), (earlier, "reject")):
            result = self.store.lookup(req)
            self.assertEqual(result.decision, decision)
            self.assertIsNone(result.refund)
        self.assertEqual(len(self.read("refunds.json")), 1)

    def test_missing_or_corrupt_state_fails_closed(self):
        pay(self.store, request(), approval())
        path = self.out / "state.json"
        snapshot = self.read("state.json")
        path.unlink()
        self.assert_code("STATE_MISSING", lambda: RefundStore(self.out, self.data))
        path.write_text("{broken", encoding="utf-8")
        self.assert_code("STATE_CORRUPT", lambda: RefundStore(self.out, self.data))
        snapshot["processed_results"].clear()
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        self.assert_code("STATE_CORRUPT", lambda: RefundStore(self.out, self.data))
        self.assertEqual(len(self.read("refunds.json")), 2)

    def test_source_hash_allows_new_requests_only(self):
        with_more = self.data.model_copy(update={"requests": [request("extra")]})
        self.assertIsNone(RefundStore(self.out, with_more).lookup(request("extra")))
        with_new_limit = self.data.model_copy(update={"meta": self.data.meta.model_copy(update={"auto_refund_limit": 10})})
        self.assert_code("SOURCE_CONFLICT", lambda: RefundStore(self.out, with_new_limit))

    def test_store_rechecks_frozen_action_and_ledger_before_write(self):
        req, a = request(), approval()
        self.assert_code("NOT_APPROVED", lambda: self.store.commit(req, a))
        self.assert_code("NOT_APPROVED", lambda: self.store.commit(req, a, create=True))
        self.assert_code("NOT_APPROVED", lambda: self.store.commit(req, declined(), create=True))
        self.assert_code("STATE_CONFLICT", lambda: self.store.commit(req, a, recheck=lambda _: approval("S-B"),
                                                                    create=True))
        self.assert_code("NOT_APPROVED", lambda: pay(self.store, req.model_copy(update={"from_email": "other"}), a))
        self.assert_code("STATE_CONFLICT", lambda: pay(self.store, req, approval("S-B", 3)))
        self.assertIsNone(self.store.lookup(req))

    def test_concurrent_processes_cannot_pay_for_last_unit_twice(self):
        ctx = multiprocessing.get_context("spawn")
        ready, decisions = ctx.Barrier(2), ctx.Queue()
        procs = [ctx.Process(target=competing_writer, args=(self.out, req, ready, decisions))
                 for req in ("first", "second")]
        try:
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(30)
                self.assertEqual(proc.exitcode, 0)
            self.assertCountEqual([decisions.get(timeout=2), decisions.get(timeout=2)], ["refund", "reject"])
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
            decisions.close()
            decisions.join_thread()
        self.assertEqual(len(self.store.refunds()), 2)
        for req in (request("first"), request("second")):
            self.assertEqual(self.store.commit(req, approval(), create=True), self.store.lookup(req))
