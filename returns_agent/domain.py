"""Deterministic interpretation of the validated P1–P10 policy."""

import calendar
import re
from collections import Counter
from datetime import date

from .contracts import (
    AgentError, Assessment, Order, Product, Refund, RefundLine, RefundRecord,
    Result, RuntimeData, SafeRequest, SemanticIntent,
)


def _returned_quantities(refunds: list[RefundRecord]) -> Counter:
    returned = Counter()
    for record in refunds:
        for line in record.lines:
            returned[record.order_id, line.sku] += line.qty
    return returned


def _scoped_orders(data: RuntimeData, safe: SafeRequest) -> list[Order]:
    return [order for order in data.orders if order.customer_id == safe.customer_id]


def case_context(data: RuntimeData, safe: SafeRequest, refunds: list[RefundRecord]) -> dict:
    products = {p.sku: p for p in data.catalog}
    returned = _returned_quantities(refunds)
    orders = []
    for order in _scoped_orders(data, safe):
        lines = []
        for line in order.lines:
            product = products[line.sku]
            remaining = line.qty - returned[order.order_id, line.sku]
            if remaining < 0:
                raise AgentError("STATE_CONFLICT")
            lines.append({
                "sku": line.sku, "name": product.name, "category": product.category,
                "qty": line.qty, "remaining_qty": remaining, "paid_unit_price": line.paid_unit_price,
                "returnable": product.returnable, "warranty_months": product.warranty_months,
            })
        orders.append({"order_id": order.order_id, "ordered_at": order.ordered_at,
                       "delivered_at": order.delivered_at, "lines": lines})
    return {
        "sanitized_text": safe.sanitized_text,
        "policy": [rule.model_dump() for rule in data.policy],
        "as_of": data.meta.as_of, "auto_refund_limit": data.meta.auto_refund_limit,
        "currency": data.meta.currency, "orders": orders,
        "payment_details_supplied": safe.payment_details_supplied,
        "destination_is_definitely_different": safe.destination_is_definitely_different,
    }


def _resolve_order(data: RuntimeData, safe: SafeRequest, intent: SemanticIntent) -> Order | None:
    """A model-selected ID must agree with the customer's scope and an unambiguous match."""
    owned = _scoped_orders(data, safe)
    owned_ids = {order.order_id for order in owned}
    if "order" in intent.uncertainties or (intent.order_id is not None and intent.order_id not in owned_ids):
        return None
    tokens = {token.casefold() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", safe.sanitized_text)}
    all_ids = {order.order_id.casefold() for order in data.orders}
    mentioned = {t for t in tokens if t in all_ids or re.fullmatch(r"o-\d+", t)}
    if mentioned:
        candidates = [o for o in owned if {o.order_id.casefold()} == mentioned]
    else:
        requested = {item.sku for item in intent.items if item.sku is not None}
        candidates = [o for o in owned if requested.issubset({line.sku for line in o.lines})]
        if len(candidates) > 1:
            candidates = _match_explicit_date(candidates, safe.sanitized_text)
    if len(candidates) != 1:
        return None
    match = candidates[0]
    return match if intent.order_id in (None, match.order_id) else None


def _match_explicit_date(candidates: list[Order], text: str) -> list[Order]:
    # A purchase/delivery date can distinguish two otherwise identical orders.
    date_mention = re.compile(
        r"\b(заказ\w*|покупк\w*|купил\w*|достав\w*|получил\w*)\s+(?:от\s+|был[аи]?\s+)?"
        r"(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})\b", re.IGNORECASE)
    for match in date_mention.finditer(text):
        label, value = match.groups()
        if "." in value:
            value = "-".join(reversed(value.split(".")))
        delivery = label.casefold().startswith(("достав", "получил"))
        candidates = [o for o in candidates if (o.delivered_at if delivery else o.ordered_at) == value]
    return candidates


_STAFF_NOTES = {
    "UNKNOWN_SENDER": "Отправитель не идентифицирован. Уточните номер заказа и контакт, использованный при покупке.",
    "AMBIGUOUS_ORDER": "Заказ не удалось однозначно сопоставить с обращением. Уточните номер заказа.",
    "AMBIGUOUS_ITEM": "Уточните у клиента наименование или SKU возвращаемого товара.",
    "AMBIGUOUS_QUANTITY": "Уточните, сколько единиц каждой позиции клиент просит вернуть.",
    "AMBIGUOUS_REASON": "Уточните, описывает ли клиент неисправность товара.",
    "AMBIGUOUS_INTENT": "Уточните, хочет ли клиент вернуть товар и какую позицию касается обращение.",
    "OTHER_REQUEST": "В обращении нет просьбы вернуть товар. Разберите вопрос клиента отдельно.",
    "ALTERNATIVE_PAYMENT": "Клиент просит перечислить деньги на другие реквизиты. Уточните, согласен ли он на возврат исходным способом.",
    "AMBIGUOUS_PAYMENT": "Уточните, имел ли клиент в виду исходный способ оплаты заказа.",
    "EXCESS_QUANTITY": "Запрос превышает остаток с учётом уже оформленных возвратов. Согласуйте с клиентом количество.",
    "DUPLICATE_ITEMS": "Одна позиция указана несколько раз. Уточните общее количество по ней.",
    "MIXED_ELIGIBILITY": "Часть позиций можно вернуть, для остальных действуют ограничения. Уточните состав возврата.",
    "DELIVERY_NOT_REACHED": "По данным заказа доставка ещё не состоялась. Уточните дату получения.",
    "PRIVACY_BLOCKED": "Не удалось безопасно подготовить данные для автоматического разбора. Проверьте обращение локально.",
    "AGENT_UNAVAILABLE": "Автоматический разбор не завершён. Проверьте обращение и условия возврата вручную.",
}
_REJECT_NOTES = {
    "ALREADY_RETURNED": "Все запрошенные единицы уже возвращены.",
    "NOT_RETURNABLE": "Эта категория товаров без брака не подлежит возврату.",
    "WINDOW_EXPIRED": "С даты доставки прошло больше 14 календарных дней. Срок возврата товара без брака истёк.",
    "WARRANTY_EXPIRED": "Гарантийный срок с даты доставки истёк.",
}


def _handoff(reason: str, rules: list, order_id: str | None = None, *, proposed: int | None = None) -> Assessment:
    return Assessment(decision="escalate", order_id=order_id, policy_rules=rules, reason_code=reason,
                      refund=None, proposed_refund_total=proposed)


def escalation(code: str) -> Assessment:
    if code == "UNKNOWN_SENDER":
        return _handoff(code, ["P8"])
    return _handoff("PRIVACY_BLOCKED" if code == "PRIVACY_BLOCKED" else "AGENT_UNAVAILABLE", [])


def _warranty_end(delivered: date, months: int) -> date:
    year_offset, zero_based_month = divmod(delivered.month - 1 + months, 12)
    year, month = delivered.year + year_offset, zero_based_month + 1
    if year > date.max.year:
        return date.max  # Every representable as_of is still in warranty.
    return date(year, month, min(delivered.day, calendar.monthrange(year, month)[1]))


def _item_eligibility(product: Product, reason: str, delivered: date, as_of: date) -> tuple[str, str | None]:
    if reason == "defect":
        expired = as_of > _warranty_end(delivered, product.warranty_months)
        return "P3", "WARRANTY_EXPIRED" if expired else None
    if not product.returnable:
        return "P2", "NOT_RETURNABLE"
    return "P1", "WINDOW_EXPIRED" if (as_of - delivered).days > 14 else None


_UNRESOLVED_FACT = {
    "intent": ("AMBIGUOUS_INTENT", ["P9"]),
    "payment": ("AMBIGUOUS_PAYMENT", ["P5"]),
    "item": ("AMBIGUOUS_ITEM", ["P8"]),
}


def assess(data: RuntimeData, safe: SafeRequest, intent: SemanticIntent,
           refunds: list[RefundRecord]) -> Assessment:
    """An assessment is a quote. Only the request-scoped create tool can commit it."""
    if any(not item.evidence.strip() or item.evidence not in safe.sanitized_text for item in intent.items):
        raise AgentError("INVALID_ARGUMENTS")
    order = _resolve_order(data, safe, intent)
    order_id = order.order_id if order else None
    if intent.alternative_payment_requested is True or safe.destination_is_definitely_different is True:
        # P10 preserves P5 even if the message purports to change the rules.
        rules = ["P5"]
        if re.search(r"\bsystem\b|игнорир\w*\s+(?:все\s+)?правил|обновлен\w*\s+политик", safe.sanitized_text, re.I):
            rules.append("P10")
        return _handoff("ALTERNATIVE_PAYMENT", rules, order_id)
    if intent.intent == "other":
        return _handoff("OTHER_REQUEST", ["P9"], order_id)
    if order is None:
        return _handoff("AMBIGUOUS_ORDER", ["P8"])
    if intent.intent == "unclear":
        return _handoff("AMBIGUOUS_INTENT", ["P9"], order_id)
    for kind, (reason, rules) in _UNRESOLVED_FACT.items():
        if kind in intent.uncertainties:
            return _handoff(reason, rules, order_id)
    if intent.alternative_payment_requested is None:
        return _handoff("AMBIGUOUS_PAYMENT", ["P5"], order_id)
    purchased = {line.sku: line for line in order.lines}
    if not intent.items or any(item.sku not in purchased for item in intent.items):
        return _handoff("AMBIGUOUS_ITEM", ["P8"], order_id)
    if len({item.sku for item in intent.items}) != len(intent.items):
        return _handoff("DUPLICATE_ITEMS", ["P7"], order_id)
    known_skus = {p.sku.casefold() for p in data.catalog}
    for item in intent.items:
        evidence_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", item.evidence)
        quoted_skus = known_skus.intersection(token.casefold() for token in evidence_tokens)
        if quoted_skus and item.sku.casefold() not in quoted_skus:
            return _handoff("AMBIGUOUS_ITEM", ["P8"], order_id)

    returned = _returned_quantities(refunds)
    available = {item.sku: purchased[item.sku].qty - returned[order_id, item.sku] for item in intent.items}
    if any(qty < 0 for qty in available.values()):
        raise AgentError("STATE_CONFLICT")
    if not any(available.values()):
        return Assessment(decision="reject", order_id=order_id, policy_rules=["P7"],
                          reason_code="ALREADY_RETURNED", refund=None, proposed_refund_total=None)
    if "quantity" in intent.uncertainties or any(item.qty is None for item in intent.items):
        return _handoff("AMBIGUOUS_QUANTITY", ["P7"], order_id)
    if any(0 < available[item.sku] < item.qty for item in intent.items):
        return _handoff("EXCESS_QUANTITY", ["P7"], order_id)
    if "reason" in intent.uncertainties or any(item.reason == "unclear" for item in intent.items):
        return _handoff("AMBIGUOUS_REASON", ["P1", "P3"], order_id)

    as_of, delivered = date.fromisoformat(data.meta.as_of), date.fromisoformat(order.delivered_at)
    if delivered > as_of:
        return _handoff("DELIVERY_NOT_REACHED", [], order_id)
    products = {p.sku: p for p in data.catalog}
    rules, failures, lines = [], [], []
    for item in intent.items:
        line = purchased[item.sku]
        if available[item.sku] == 0:
            rules.append("P7")
            failures.append("ALREADY_RETURNED")
            continue
        rule, failure = _item_eligibility(products[item.sku], item.reason, delivered, as_of)
        rules.append(rule)
        if failure:
            failures.append(failure)
        else:
            lines.append(RefundLine(sku=line.sku, qty=item.qty, amount=line.paid_unit_price * item.qty))
    rules = list(dict.fromkeys(rules))
    if failures:
        if lines:
            return _handoff("MIXED_ELIGIBILITY", rules, order_id)
        reason = failures[0] if len(set(failures)) == 1 else "MULTIPLE_REFUSALS"
        return Assessment(decision="reject", order_id=order_id, policy_rules=rules,
                          reason_code=reason, refund=None, proposed_refund_total=None)
    total = sum(line.amount for line in lines)
    rules.extend(["P4", "P5", "P6", "P7"])
    if total > data.meta.auto_refund_limit:
        return _handoff("OVER_LIMIT", rules, order_id, proposed=total)
    quote = Refund(lines=sorted(lines, key=lambda line: line.sku), total=total, method=order.payment_method)
    return Assessment(decision="refund", order_id=order_id, policy_rules=rules, reason_code="APPROVED",
                      refund=quote, proposed_refund_total=None)


def build_result(request_id: str, assessment: Assessment, confirmed: RefundRecord | None = None) -> Result:
    refund, message, staff = None, None, None
    if assessment.decision == "refund":
        if confirmed is None or assessment.refund is None:
            raise AgentError("NOT_APPROVED")
        refund = Refund(lines=confirmed.lines, total=confirmed.total, method=confirmed.method)
        if (confirmed.request_id != request_id or confirmed.order_id != assessment.order_id
                or refund != assessment.refund):
            raise AgentError("STATE_CONFLICT")
        message = f"По заказу {confirmed.order_id} оформлен возврат {refund.total} ₽ на исходный способ оплаты."
    elif confirmed is not None:
        raise AgentError("STATE_CONFLICT")
    elif assessment.decision == "reject":
        if assessment.reason_code == "MULTIPLE_REFUSALS":
            reasons = {"P1": "истёк срок возврата без брака", "P2": "товар без брака невозвратный",
                       "P3": "истекла гарантия", "P7": "позиция уже возвращена"}
            detail = ", ".join(reasons[r] for r in assessment.policy_rules if r in reasons)
            message = f"Возврат по заказу {assessment.order_id} отклонён. Причины по запрошенным позициям — {detail}."
        else:
            message = _REJECT_NOTES.get(assessment.reason_code)
            if message is None:
                raise AgentError("INVALID_ASSESSMENT")
    elif assessment.reason_code == "OVER_LIMIT":
        staff = (f"Расчёт по цене заказа — {assessment.proposed_refund_total} ₽. "
                 "Сумма превышает лимит автоматического возврата. Передайте на согласование.")
    else:
        staff = _STAFF_NOTES.get(assessment.reason_code)
        if staff is None:
            raise AgentError("INVALID_ASSESSMENT")
    return Result(request_id=request_id, decision=assessment.decision, order_id=assessment.order_id,
                  policy_rules=assessment.policy_rules, refund=refund,
                  proposed_refund_total=assessment.proposed_refund_total,
                  message_to_customer=message, note_for_staff=staff)
