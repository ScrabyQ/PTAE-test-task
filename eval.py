"""Independent scoring and ledger checks. Only this entry point reads golden."""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from returns_agent.contracts import AgentError, RefundRecord, Result
from returns_agent.data import DEFAULT_DATA, load_runtime


MISSING = "<missing>"
INVALID = "<invalid>"


def _field(value, key):
    if isinstance(value, dict):
        return value.get(key, MISSING)
    return INVALID


def _printable(value):
    # A malformed file can contain arbitrary private prose. Show only comparison-field values.
    if value in (MISSING, INVALID) or value is None or type(value) in (int, bool, float):
        return value
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", value):
        return value
    return INVALID


def projection(row, *, reference=False):
    refund = _field(row, "refund")
    if refund is None:
        lines, total = None, None
    elif isinstance(refund, dict):
        lines, total = _field(refund, "lines"), _printable(_field(refund, "total"))
        if isinstance(lines, list):
            lines = sorted(
                [{k: _printable(_field(line, k)) for k in ("sku", "qty", "amount")} for line in lines],
                key=lambda line: json.dumps(line, sort_keys=True),
            )
        else:
            lines = _printable(lines)
    else:
        lines = total = MISSING if refund == MISSING else INVALID
    proposed = _field(row, "proposed_refund_total")
    return {"decision": _printable(_field(row, "decision")), "order_id": _printable(_field(row, "order_id")),
            "refund.lines": lines, "refund.total": total,
            "proposed_refund_total": None if reference and proposed == MISSING else _printable(proposed)}


def score(expected_rows, result_rows):
    matched, differences = 0, []
    for expected in expected_rows:
        rid = expected["request_id"]
        hits = [r for r in result_rows if isinstance(r, dict) and r.get("request_id") == rid]
        if len(hits) != 1:
            differences.append((rid, "results.count", 1, len(hits)))
            continue
        expected_fields, actual = projection(expected, reference=True), projection(hits[0])
        changed = []
        for field, value in expected_fields.items():
            # JSON preserves distinctions such as true vs 1.
            if json.dumps(value, sort_keys=True) != json.dumps(actual[field], sort_keys=True):
                changed.append((rid, field, value, actual[field]))
        differences.extend(changed)
        matched += not changed
    return matched, differences


def check_ledger(data, rows, refunds_raw):
    issues = []
    ordered = sorted(data.requests, key=lambda r: datetime.fromisoformat(r.received_at))
    requests = {r.request_id: r for r in ordered}
    ids = [_field(row, "request_id") for row in rows]
    if ids != list(requests):
        issues.append("Состав, уникальность или порядок request_id расходятся с requests по received_at.")
    orders = {o.order_id: o for o in data.orders}
    customers = {c.email.casefold().strip(): c.customer_id for c in data.customers}
    legacy = {r.refund_id: r for r in data.refunds}
    if not isinstance(refunds_raw, list):
        return issues + ["Ожидался массив записей в refunds.json."]
    try:
        exported = [RefundRecord.model_validate(raw) for raw in refunds_raw]
    except (ValidationError, TypeError):
        return issues + ["Структура refunds.json повреждена."]
    by_id = {r.refund_id: r for r in exported}
    if len(exported) != len(by_id) or any(by_id.get(key) != old for key, old in legacy.items()):
        issues.append("Не совпадают идентификаторы возвратов или исходные записи legacy.")
    records = {r.request_id: r for r in exported if r.refund_id not in legacy}
    if len({r.request_id for r in exported}) != len(exported):
        issues.append("Один request_id встречается в нескольких записях refunds.json.")
    quantities = Counter()
    for record in data.refunds:
        for line in record.lines:
            quantities[record.order_id, line.sku] += line.qty
    processed_refunds = set()
    for row in rows:
        rid = _field(row, "request_id")
        label = rid if isinstance(rid, str) and rid in requests else INVALID
        try:
            result = Result.model_validate(row)
        except ValidationError:
            issues.append(f"{label} — нарушение схемы результата или обязательных null.")
            continue
        order = orders.get(result.order_id)
        if result.order_id is not None and (order is None or rid not in requests or
                customers.get(requests[rid].from_email.casefold().strip()) != order.customer_id):
            issues.append(f"{label} — заказ вне области отправителя.")
        if result.proposed_refund_total is not None and result.proposed_refund_total <= data.meta.auto_refund_limit:
            issues.append(f"{label} — сумма для согласования не превышает лимит.")
        if result.decision != "refund":
            continue
        processed_refunds.add(rid)
        refund = result.refund
        if order is None:
            continue
        if refund.method != order.payment_method:
            issues.append(f"{label} — refund.method отличается от исходного способа оплаты.")
        if refund.total > data.meta.auto_refund_limit:
            issues.append(f"{label} — превышен лимит автоматического возврата.")
        purchased = {line.sku: line for line in order.lines}
        for line in refund.lines:
            key = order.order_id, line.sku
            quantities[key] += line.qty
            item = purchased.get(line.sku)
            if item is None or quantities[key] > item.qty or line.amount != item.paid_unit_price * line.qty:
                issues.append(f"{label} — позиция, сумма или остаток расходятся с заказом.")
        record = records.get(rid)
        if (record is None or record.order_id != order.order_id or record.lines != refund.lines or
                record.total != refund.total or record.method != refund.method):
            issues.append(f"{label} — результат расходится с операцией в refunds.json.")
    if processed_refunds != set(records):
        issues.append("Набор новых операций расходится с решениями refund.")
    return issues


def evaluate_files(data_path: Path, out: Path) -> int:
    try:
        data = load_runtime(data_path)
        golden = json.loads(data_path.read_text(encoding="utf-8-sig")).get("golden", [])
        rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not isinstance(golden, list):
            raise ValueError
        known_ids = {r.request_id for r in data.requests}
        expected_ids = [r["request_id"] for r in golden]
        if len(expected_ids) != len(set(expected_ids)) or not set(expected_ids) <= known_ids:
            raise ValueError
        matched, differences = score(golden, rows)
        print(f"Eval: совпало {matched}/{len(golden)}." if golden else "В наборе нет golden. Основной score не вычисляется.")
        for rid, field, expected, actual in differences:
            print(f"{rid} {field}: expected={json.dumps(expected, ensure_ascii=False)}, "
                  f"actual={json.dumps(actual, ensure_ascii=False)}")
        refunds_raw = json.loads((out / "refunds.json").read_text(encoding="utf-8"))
        issues = check_ledger(data, rows, refunds_raw)
        for issue in issues:
            print(f"Инвариант: {issue}")
        if not issues:
            print(f"Финансовые инварианты и выгрузки проверены для {len(rows)} решений.")
        return int(bool(differences or issues))
    except (AgentError, OSError, ValueError, TypeError, KeyError):
        print("Eval: входные данные или обязательные выгрузки отсутствуют либо повреждены.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Сравнить результаты с golden и проверить реестр возвратов.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    raise SystemExit(evaluate_files(args.data, args.out))
