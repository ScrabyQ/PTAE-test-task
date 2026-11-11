import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from returns_agent.config import BASE_URL, MODEL_ALIAS, Settings
from returns_agent.contracts import AgentError
from returns_agent.data import DEFAULT_DATA, load_runtime
from returns_agent.runner import process_batch


OFFLINE = Settings(BASE_URL, MODEL_ALIAS, "offline-secret-canary")
NO_KEY = Settings(BASE_URL, MODEL_ALIAS, "")


def function_call(name, args):
    return {"type": "function", "id": "provider-id", "function": {
        "name": name, "arguments": json.dumps(args, ensure_ascii=False),
    }}


def facts(order_id=None, sku=None, evidence=None):
    return {"intent": {
        "intent": "return_goods" if sku else "other", "order_id": order_id,
        "items": [{"sku": sku, "qty": 1, "reason": "no_defect", "evidence": evidence}] if sku else [],
        "alternative_payment_requested": False, "uncertainties": [],
    }}


class Wire:
    def __init__(self, respond):
        self.respond = respond
        self.payloads = []

    def handle(self, request):
        assert str(request.url) == BASE_URL + "/chat/completions"
        body = json.loads(request.content)
        self.payloads.append(body)
        stage = body["tool_choice"]["function"]["name"]
        return httpx.Response(200, json={
            "model": "deepseek/offline-test", "choices": [{"message": {
                "content": "discarded-provider-prose", "tool_calls": self.respond(stage, body),
            }}],
        })


class ControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_runtime(DEFAULT_DATA)
        cls.order = next(o for o in cls.data.orders if any(l.sku == "KTL-200" for l in o.lines))
        cls.sender = next(c.email for c in cls.data.customers if c.customer_id == cls.order.customer_id)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "new"

    def paraphrase(self):
        return self.data.requests[0].model_copy(update={
            "request_id": "new-example", "from_email": self.sender,
            "text": f"Заказ {self.order.order_id}. Передумал, примите обратно один KTL-200.",
        })

    def synthetic(self, respond):
        dataset = self.data.model_copy(update={"requests": [self.paraphrase()]})
        wire = Wire(respond)
        summary = process_batch(dataset, self.out, OFFLINE, transport=httpx.MockTransport(wire.handle))
        result = json.loads((self.out / "results.json").read_text(encoding="utf8"))[0]
        return dataset, summary, result, wire

    def test_refund_first_run_and_keyless_replay_preserve_bytes_and_traces(self):
        intent = facts(self.order.order_id, "KTL-200", "один KTL-200")
        def respond(stage, body):
            return [function_call(stage, intent if stage == "assess_return" else {})]
        dataset, summary, result, wire = self.synthetic(respond)
        self.assertEqual(summary["http_requests"], 3)
        self.assertEqual(result["decision"], "refund")
        self.assertEqual(result["refund"]["method"], self.order.payment_method)
        files = (self.out / "state.json", self.out / "refunds.json", self.out / "results.json",
                 *list((self.out / "traces" / summary["run_id"]).glob("*.jsonl")))
        before = [p.read_bytes() for p in files]
        with patch("returns_agent.runner.PrivacyGate", side_effect=AssertionError("replay initialized NLP")):
            replay = process_batch(dataset, self.out, NO_KEY)
        self.assertEqual((replay["http_requests"], replay["cached"]), (0, 1))
        self.assertEqual(before, [p.read_bytes() for p in files])
        audit = b"".join(before)
        for needle in ("offline-secret-canary", "discarded-provider-prose", self.sender, self.paraphrase().text):
            self.assertFalse(needle.encode() in audit)
        self.assertNotIn("reasoning", json.dumps(wire.payloads))

    def test_entire_corpus_actual_wire_projection(self):
        wire = Wire(lambda stage, body: [function_call(stage, facts() if stage == "assess_return" else {})])
        summary = process_batch(self.data, self.out, OFFLINE, transport=httpx.MockTransport(wire.handle))
        known = {c.email.casefold() for c in self.data.customers}
        expected = sum(r.from_email.casefold() in known for r in self.data.requests)
        self.assertEqual(summary["processed"], len(self.data.requests))
        self.assertEqual(len(wire.payloads), 2 * expected)
        sent = json.dumps(wire.payloads, ensure_ascii=False).casefold()
        audit = " ".join(p.read_text(encoding="utf8") for p in self.out.rglob("*.json*")).casefold()
        for corpus in (sent, audit):
            digits = re.sub(r"[\s()+.\-]", "", corpus)
            for c in self.data.customers:
                private = [c.email, c.address, *c.full_name.split()]
                self.assertFalse(any(part.casefold() in corpus for part in private))
                self.assertFalse(re.sub(r"\D", "", c.phone)[-10:] in digits)
            for req in self.data.requests:
                self.assertFalse(req.from_email.casefold() in corpus)
                for pan in re.findall(r"(?<!\d)\d{4}(?:[ -]?\d{4}){3}(?!\d)", req.text):
                    self.assertFalse(re.sub(r"\D", "", pan) in digits)
            self.assertFalse(any(s in corpus for s in ("offline-secret-canary", "discarded-provider-prose")))
        self.assertFalse(any(k in sent for k in ('"customers"', '"from_email"', '"golden"', '"plugins"')))
        self.assertEqual(len(list(self.out.rglob("case_*.jsonl"))), len(self.data.requests))

    def test_extra_amount_or_bool_quantity_gets_only_one_repair(self):
        invalid = facts(self.order.order_id, "KTL-200", "один KTL-200")
        invalid["intent"]["items"][0]["qty"] = True
        invalid["intent"]["amount"] = 999999
        _, summary, result, _ = self.synthetic(lambda stage, _: [
            function_call(stage, invalid if stage == "assess_return" else {})
        ])
        self.assertEqual(summary["http_requests"], 3)
        self.assertEqual(result["decision"], "escalate")
        self.assertIsNone(result["refund"])
        self.assertEqual(result["policy_rules"], [])

    def test_injected_early_or_parallel_create_never_reaches_store(self):
        for count in (1, 2, 7):
            self.out = self.out / f"separate-{count}"
            _, summary, result, _ = self.synthetic(lambda *_: [function_call("create_refund", {})] * count)
            self.assertEqual(summary["http_requests"], 1)
            self.assertEqual(result["decision"], "escalate")
            self.assertEqual(len(json.loads((self.out / "refunds.json").read_text())), len(self.data.refunds))

    def test_unavailable_api_and_missing_key_do_not_create_case_decisions(self):
        dataset = self.data.model_copy(update={"requests": [self.paraphrase()]})
        with self.assertRaisesRegex(AgentError, "^ROUTERAI_KEY_MISSING$"):
            process_batch(dataset, self.out, NO_KEY)
        self.assertFalse(self.out.exists())
        transport = httpx.MockTransport(lambda _: httpx.Response(401, text="offline-secret-canary"))
        with self.assertRaisesRegex(AgentError, "^ROUTERAI_HTTP_401$"):
            process_batch(dataset, self.out, OFFLINE, transport=transport)
        self.assertEqual(json.loads((self.out / "results.json").read_text()), [])

    def test_connection_timeout_is_a_batch_failure_after_one_retry(self):
        dataset = self.data.model_copy(update={"requests": [self.paraphrase()]})
        attempts = []

        def unavailable(request):
            attempts.append(request.url.host)
            raise httpx.ConnectTimeout("test-only", request=request)

        with patch("returns_agent.routerai.time.sleep"), self.assertRaisesRegex(AgentError, "^ROUTERAI_UNAVAILABLE$"):
            process_batch(dataset, self.out, OFFLINE, transport=httpx.MockTransport(unavailable))
        self.assertEqual(attempts, ["routerai.ru", "routerai.ru"])
        self.assertEqual(json.loads((self.out / "results.json").read_text()), [])
