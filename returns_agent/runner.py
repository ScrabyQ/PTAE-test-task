import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentController, RequestContext, public_result
from .config import Settings
from .contracts import AgentError, PrivacyEscalation
from .data import load_runtime
from .domain import escalation
from .privacy import PrivacyGate
from .routerai import RouterAI
from .store import RefundStore


class Trace:
    def __init__(self, directory, run_id, request_id, mode):
        self.file = directory / run_id / f"case_{request_id}.jsonl"
        self.run_id, self.request_id, self.mode = run_id, request_id, mode
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.touch(exist_ok=False)
        except OSError:
            raise AgentError("TRACE_WRITE_FAILED") from None

    def __call__(self, event, **fields):
        row = {"at": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
               "request_id": self.request_id, "mode": self.mode, "event": event, **fields}
        try:
            with self.file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except OSError:
            raise AgentError("TRACE_WRITE_FAILED") from None


def process_batch(data, out: Path, settings: Settings, *, transport=None):
    # On an unused directory, report missing credentials before creating submission artifacts.
    known = {c.email.strip().casefold() for c in data.customers}
    history = any((out / name).exists() for name in ("state.json", "refunds.json", "results.json", "traces"))
    if not history and any(r.from_email.strip().casefold() in known for r in data.requests):
        settings.require_access()
    store = RefundStore(out, data)
    counts, cached = Counter(), 0
    run_id = uuid.uuid4().hex
    gate, router = None, None
    try:
        for request in sorted(data.requests, key=lambda r: datetime.fromisoformat(r.received_at)):
            existing = store.lookup(request)
            mode = "test_transport" if transport is not None else "routerai"
            trace = Trace(out / "traces", run_id, request.request_id, mode)
            if existing is not None:
                result = existing
                trace("replay", result=public_result(result))
                cached += 1
            else:
                if gate is None:
                    gate = PrivacyGate(data)
                safe = gate.prepare(request)
                if isinstance(safe, PrivacyEscalation):
                    trace("local_handoff", code=safe.code)
                    result = store.commit(request, escalation(safe.code))
                else:
                    trace("prepared", privacy_version=safe.privacy_version, redactions=safe.redactions)
                    if router is None:
                        router = RouterAI(settings, gate, transport)
                    result = AgentController(data, store, router).process(RequestContext(request, safe), trace)
                trace("committed_result", result=public_result(result))
            counts[result.decision] += 1
        return {"run_id": run_id, "processed": sum(counts.values()), "decisions": dict(counts),
                "cached": cached, "http_requests": router.http_requests if router else 0}
    finally:
        if router is not None:
            router.close()


def run(data_path: Path, out: Path) -> None:
    data = load_runtime(data_path)
    summary = process_batch(data, out, Settings.from_env())
    print(f"Обработано {summary['processed']}, из состояния {summary['cached']}, "
          f"HTTP-запросов к RouterAI {summary['http_requests']}. Решения {summary['decisions']}.")
    print(f"Выгрузки и состояние в {out.resolve()}. Трассы прогона {summary['run_id']}.", flush=True)
