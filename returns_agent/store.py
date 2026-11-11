"""One local ledger. Every writer re-reads and publishes under the same file lock."""

import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from filelock import FileLock, Timeout

from .contracts import (
    AgentError, Assessment, PrivateRequest, Processed, RefundRecord, Result, RuntimeData, State, digest,
)
from .domain import build_result


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_replace(path: Path, content: bytes) -> None:
    # Windows requires the temporary file to be closed before replacement.
    temp = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with temp:
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp.name, path)
    except BaseException:
        # A failed write may leave a partial temp file. After a successful rename
        # there is nothing left to unlink and no operation left that can fail here.
        try:
            Path(temp.name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _unique_keys(pairs: list[tuple]) -> dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("DUPLICATE_STATE_KEY")
    return obj


def _action_hash(action: Assessment | Result) -> str:
    # The order of independently requested SKUs has no financial significance.
    charge = action.refund.model_dump() if action.refund is not None else None
    if charge is not None:
        charge["lines"].sort(key=lambda line: line["sku"])
    return digest({
        "decision": action.decision, "order_id": action.order_id, "refund": charge,
        "proposed_refund_total": action.proposed_refund_total,
    })


class RefundStore:
    def __init__(self, out: Path, data: RuntimeData):
        self.out = Path(out)
        self.path = self.out / "state.json"
        self.lock = FileLock(self.out / ".state.lock", timeout=10)
        self.source_hash = digest(data.model_dump(exclude={"requests"}))
        self.legacy = data.refunds
        self.legacy_requests = {record.request_id for record in self.legacy}
        self.orders = {order.order_id: order for order in data.orders}
        self.purchased = {(order.order_id, line.sku): line for order in data.orders for line in order.lines}
        self.customers = {c.email.strip().casefold(): c.customer_id for c in data.customers}
        self.limit = data.meta.auto_refund_limit
        self.as_of = data.meta.as_of
        try:
            self.out.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise AgentError("STATE_UNAVAILABLE") from None
        with self._locked():
            if self.path.exists():
                state = self._read()
            else:
                traces = self.out / "traces"
                if (self.out / "refunds.json").exists() or (self.out / "results.json").exists() or (
                    traces.exists() and any(traces.iterdir())
                ):
                    raise AgentError("STATE_MISSING")
                state = State(schema_version=1, source_hash=self.source_hash, refunds=self.legacy, processed_results={})
                self._write_state(state)
            self._export(state)

    @contextmanager
    def _locked(self):
        try:
            with self.lock:
                yield
        except Timeout:
            raise AgentError("STATE_LOCK_TIMEOUT") from None
        except OSError:
            raise AgentError("STATE_UNAVAILABLE") from None

    def _read(self) -> State:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
            state = State.model_validate(raw)
            if state.source_hash != self.source_hash:
                raise AgentError("SOURCE_CONFLICT")
            self._verify(state)
            return state
        except FileNotFoundError:
            raise AgentError("STATE_MISSING") from None
        except (ValueError, KeyError, TypeError, OverflowError):
            raise AgentError("STATE_CORRUPT") from None

    def _verify(self, state: State) -> None:
        if state.refunds[:len(self.legacy)] != self.legacy:
            raise ValueError("MISSING_LEGACY_REFUND")
        issued = {r.request_id: r for r in state.refunds}
        if len(issued) != len(state.refunds) or len({r.refund_id for r in state.refunds}) != len(issued):
            raise ValueError("DUPLICATE_REFUND")
        returned = Counter()
        for record in state.refunds:
            if self.orders[record.order_id].payment_method != record.method:
                raise ValueError("WRONG_METHOD")
            for line in record.lines:
                key = record.order_id, line.sku
                purchased = self.purchased[key]
                returned[key] += line.qty
                if returned[key] > purchased.qty or line.amount != purchased.paid_unit_price * line.qty:
                    raise ValueError("INVALID_REFUND_LINE")

        new_refunds = set(issued) - self.legacy_requests
        paid = set()
        for key, processed in state.processed_results.items():
            result = processed.result
            if result.request_id != key or key in self.legacy_requests:
                raise ValueError("INVALID_RESULT_KEY")
            if datetime.fromisoformat(processed.received_at).utcoffset() is None:
                raise ValueError("INVALID_TIMESTAMP")
            if result.order_id is not None and result.order_id not in self.orders:
                raise ValueError("UNKNOWN_RESULT_ORDER")
            if result.proposed_refund_total is not None and result.proposed_refund_total <= self.limit:
                raise ValueError("INVALID_PROPOSAL")
            if result.decision == "refund":
                record = issued[key]
                if (record.order_id != result.order_id or record.lines != result.refund.lines
                        or record.total != result.refund.total or record.method != result.refund.method
                        or record.total > self.limit or record.created_at != self.as_of
                        or processed.action_hash != _action_hash(result)):
                    raise ValueError("MISMATCHED_REFUND_RESULT")
                paid.add(key)
        if paid != new_refunds:
            raise ValueError("MISSING_REFUND_RESULT")

    def _write_state(self, state: State) -> None:
        try:
            _atomic_replace(self.path, _json_bytes(state.model_dump()))
        except OSError:
            raise AgentError("STATE_WRITE_FAILED") from None

    def _export(self, state: State) -> None:
        ordered = sorted(state.processed_results.values(), key=lambda p: datetime.fromisoformat(p.received_at))
        exports = (
            ("refunds.json", [r.model_dump() for r in state.refunds]),
            ("results.json", [p.result.model_dump() for p in ordered]),
        )
        try:
            for name, values in exports:
                path = self.out / name
                content = _json_bytes(values)
                if not path.exists() or path.read_bytes() != content:
                    _atomic_replace(path, content)
        except OSError:
            # The ledger already exists. Only the projections need to be published again.
            raise AgentError("EXPORT_FAILED_COMMITTED", committed=True) from None

    def _existing(self, state: State, request_id: str, request_hash: str) -> Processed | None:
        processed = state.processed_results.get(request_id)
        if request_id in self.legacy_requests or (processed is not None and processed.request_hash != request_hash):
            raise AgentError("IDEMPOTENCY_CONFLICT")
        return processed

    def lookup(self, req: PrivateRequest) -> Result | None:
        request_hash = digest(req.model_dump())
        with self._locked():
            state = self._read()
            existing = self._existing(state, req.request_id, request_hash)
            self._export(state)
            return existing.result if existing else None

    def refunds(self) -> list[RefundRecord]:
        with self._locked():
            return self._read().refunds

    def _record(self, req: PrivateRequest, approved: Assessment, state: State) -> RefundRecord:
        order = self.orders.get(approved.order_id)
        if (approved.refund is None or order is None
                or self.customers.get(req.from_email.strip().casefold()) != order.customer_id):
            raise AgentError("NOT_APPROVED")
        ids = {record.refund_id for record in state.refunds}
        number = len(ids) + 1
        while f"RF-{number:04d}" in ids:
            number += 1
        return RefundRecord(
            refund_id=f"RF-{number:04d}", order_id=order.order_id, request_id=req.request_id,
            created_at=self.as_of, **approved.refund.model_dump(),
        )

    def commit(
        self, req: PrivateRequest, assessment: Assessment, *,
        recheck: Callable[[list[RefundRecord]], Assessment] | None = None, create: bool = False,
    ) -> Result:
        request_hash = digest(req.model_dump())
        action_hash = _action_hash(assessment)
        with self._locked():
            state = self._read()
            previous = self._existing(state, req.request_id, request_hash)
            if create != (assessment.decision == "refund"):
                raise AgentError("NOT_APPROVED")
            if previous is not None:
                if previous.action_hash != action_hash:
                    raise AgentError("IDEMPOTENCY_CONFLICT")
                self._export(state)
                return previous.result
            if create and recheck is None:
                raise AgentError("NOT_APPROVED")

            # recheck is the pure domain evaluation of the same frozen intent, without NLP or network.
            final = recheck(list(state.refunds)) if recheck is not None else assessment
            record = None
            if final.decision == "refund":
                if not create or _action_hash(final) != action_hash:
                    raise AgentError("STATE_CONFLICT")
                record = self._record(req, final, state)
            result = build_result(req.request_id, final, record)
            entry = Processed(
                request_hash=request_hash, action_hash=action_hash, received_at=req.received_at, result=result,
            )
            updated = state.model_copy(update={
                "refunds": [*state.refunds, record] if record is not None else state.refunds,
                "processed_results": {**state.processed_results, req.request_id: entry},
            })
            try:
                self._verify(updated)
            except (ValueError, KeyError, TypeError, OverflowError):
                raise AgentError("STATE_CONFLICT") from None
            self._write_state(updated)  # Single commit point for the operation and its final result.
            self._export(updated)
            return result
