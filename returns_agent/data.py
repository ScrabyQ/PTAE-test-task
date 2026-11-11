import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

from .contracts import AgentError, RuntimeData, digest


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "task_requirements" / "data.json"
POLICY_V1 = "1b4527ca67a8db40236d8ca8916426eb98b7e6b74864a369cca2e289f1cff2d5"


def _unique(values):
    values = list(values)
    if len(set(values)) != len(values):
        raise ValueError("DUPLICATE_SOURCE_ID")


def load_runtime(path: Path) -> RuntimeData:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        data = RuntimeData.model_validate({name: raw[name] for name in RuntimeData.model_fields})
        del raw
        if digest([rule.model_dump() for rule in data.policy]) != POLICY_V1:
            raise AgentError("UNSUPPORTED_POLICY")
        date.fromisoformat(data.meta.as_of)
        _unique(c.customer_id for c in data.customers)
        _unique(c.email.strip().casefold() for c in data.customers)
        _unique(o.order_id for o in data.orders)
        _unique(p.sku for p in data.catalog)
        _unique(r.request_id for r in data.requests)
        _unique(r.refund_id for r in data.refunds)
        _unique(r.request_id for r in data.refunds)
        catalog = {p.sku for p in data.catalog}
        customers = {c.customer_id for c in data.customers}
        orders = {o.order_id: o for o in data.orders}
        purchased, previously_returned = {}, Counter()
        for order in data.orders:
            _unique(line.sku for line in order.lines)
            if order.customer_id not in customers or date.fromisoformat(order.ordered_at) > date.fromisoformat(order.delivered_at):
                raise ValueError("INVALID_ORDER")
            # Only a payment method label or the source's masked card may be persisted.
            if not re.fullmatch(r"(?:[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё _-]{0,35}|card \*{4} [0-9]{4})", order.payment_method):
                raise ValueError("INVALID_PAYMENT_LABEL")
            for line in order.lines:
                if line.sku not in catalog:
                    raise ValueError("MISSING_PRODUCT")
                purchased[order.order_id, line.sku] = line
        for refund in data.refunds:
            order = orders[refund.order_id]
            if refund.method != order.payment_method:
                raise ValueError("INVALID_LEGACY_METHOD")
            date.fromisoformat(refund.created_at)
            for line in refund.lines:
                key = (order.order_id, line.sku)
                previously_returned[key] += line.qty
                if previously_returned[key] > purchased[key].qty or line.amount != line.qty * purchased[key].paid_unit_price:
                    raise ValueError("INVALID_LEGACY_LINE")
        for req in data.requests:
            if datetime.fromisoformat(req.received_at).utcoffset() is None:
                raise ValueError("TIMEZONE_REQUIRED")
        return data
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, ValidationError):
        raise AgentError("INVALID_SOURCE_DATA") from None
