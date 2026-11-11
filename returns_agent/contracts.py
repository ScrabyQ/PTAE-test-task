"""Shared input, tool, and persistence contracts. Private objects stay on the host."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Qty = Annotated[int, Field(gt=0)]
Rub = Annotated[int, Field(ge=0)]  # The source contract uses whole RUB.
Decision = Literal["refund", "reject", "escalate"]
RuleId = Literal["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
# A bounded vocabulary keeps model-generated explanations out of state and logs.
Uncertainty = Literal["order", "item", "quantity", "reason", "payment", "intent"]


class AgentError(Exception):
    """Only fixed host-defined codes may cross into logs or CLI output."""

    def __init__(self, code: str, *, committed: bool = False):
        self.code = code
        self.committed = committed
        super().__init__(code)


class Meta(Contract):
    description: str
    as_of: str
    timezone: str
    currency: Literal["RUB"]
    auto_refund_limit: Rub


class PolicyRule(Contract):
    id: RuleId
    title: str
    text: str


class Product(Contract):
    sku: Id
    name: str
    category: str
    price: Rub
    warranty_months: Annotated[int, Field(ge=0)]
    returnable: bool


class Customer(Contract):
    customer_id: Id
    full_name: str
    email: str
    phone: str
    address: str


class OrderLine(Contract):
    sku: Id
    qty: Qty
    paid_unit_price: Rub


class Order(Contract):
    order_id: Id
    customer_id: Id
    ordered_at: str
    delivered_at: str
    payment_method: str
    delivery_cost: Rub
    promo_code: str | None = None
    lines: Annotated[list[OrderLine], Field(min_length=1)]


class PrivateRequest(Contract):
    request_id: Id
    received_at: str
    channel: str
    from_email: str
    text: str


class RefundLine(Contract):
    sku: Id
    qty: Qty
    amount: Rub


class Refund(Contract):
    lines: Annotated[list[RefundLine], Field(min_length=1)]
    total: Rub
    method: str

    @model_validator(mode="after")
    def consistent_lines(self):
        if sum(line.amount for line in self.lines) != self.total:
            raise ValueError("INVALID_TOTAL")
        if len({line.sku for line in self.lines}) != len(self.lines):
            raise ValueError("DUPLICATE_REFUND_SKU")
        return self


class RefundRecord(Refund):
    refund_id: Id
    order_id: Id
    request_id: Id
    created_at: str


class RuntimeData(Contract):
    meta: Meta
    policy: list[PolicyRule]
    catalog: list[Product]
    customers: list[Customer]
    orders: list[Order]
    refunds: list[RefundRecord]
    requests: list[PrivateRequest]


class SafeRequest(Contract):
    request_id: Id
    customer_id: Id  # internal scope; never include the customer record in a tool result
    sanitized_text: str
    privacy_version: str
    redactions: dict[str, int]
    payment_details_supplied: bool
    destination_is_definitely_different: bool | None


class PrivacyEscalation(Contract):
    code: Literal["UNKNOWN_SENDER", "PRIVACY_BLOCKED"]


class IntentItem(Contract):
    sku: Id | None
    qty: Qty | None
    reason: Literal["defect", "no_defect", "unclear"]
    evidence: Annotated[str, Field(min_length=1, max_length=350)]


class SemanticIntent(Contract):
    intent: Literal["return_goods", "other", "unclear"]
    order_id: Id | None
    items: Annotated[list[IntentItem], Field(max_length=32)]
    alternative_payment_requested: bool | None
    uncertainties: list[Uncertainty]


class NoArgs(Contract):
    pass


class AssessArgs(Contract):
    intent: SemanticIntent


TOOL_ARGS = {"get_case_context": NoArgs, "assess_return": AssessArgs, "create_refund": NoArgs}
TOOL_DESCRIPTIONS = {
    "get_case_context": "Получить обезличенное обращение, разрешённые заказы и политику.",
    "assess_return": "Передать смысловые факты для проверки policy. Для неизвестных значений использовать null.",
    "create_refund": "Зафиксировать ранее разрешённый возврат текущего обращения на исходный способ оплаты.",
}


def tool_schema(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": TOOL_DESCRIPTIONS[name],
        "parameters": TOOL_ARGS[name].model_json_schema(), "strict": True,
    }}


def parse_tool_args(name: str, encoded: str) -> Contract:
    def unique_keys(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("DUPLICATE_KEY")
        return obj

    try:
        if not isinstance(encoded, str) or len(encoded) > 24_000:
            raise ValueError("INVALID_ARGUMENTS")
        return TOOL_ARGS[name].model_validate(json.loads(encoded, object_pairs_hook=unique_keys))
    except (ValueError, TypeError, KeyError, RecursionError, ValidationError):
        raise AgentError("INVALID_ARGUMENTS") from None


class Assessment(Contract):
    decision: Decision
    order_id: Id | None
    policy_rules: list[RuleId]
    reason_code: str
    refund: Refund | None
    proposed_refund_total: Rub | None


class Result(Contract):
    request_id: Id
    decision: Decision
    order_id: Id | None
    policy_rules: list[RuleId]
    refund: Refund | None
    proposed_refund_total: Rub | None
    message_to_customer: str | None
    note_for_staff: str | None

    @model_validator(mode="after")
    def consistent_outcome(self):
        paying = self.decision == "refund"
        escalate = self.decision == "escalate"
        if paying != (self.refund is not None) or (paying and self.order_id is None):
            raise ValueError("INVALID_REFUND")
        if self.proposed_refund_total is not None and not (escalate and "P6" in self.policy_rules):
            raise ValueError("INVALID_PROPOSAL")
        if escalate:
            if not self.note_for_staff or self.message_to_customer is not None:
                raise ValueError("INVALID_STAFF_NOTE")
        elif not self.message_to_customer or self.note_for_staff is not None:
            raise ValueError("INVALID_CUSTOMER_MESSAGE")
        return self


class Processed(Contract):
    request_hash: Hash
    action_hash: Hash
    received_at: str
    result: Result


class State(Contract):
    schema_version: Literal[1]
    source_hash: Hash
    refunds: list[RefundRecord]
    processed_results: dict[Id, Processed]


def digest(value: object) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
