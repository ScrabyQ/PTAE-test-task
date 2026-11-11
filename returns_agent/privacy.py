"""Local redaction and inspection of the actual Chat Completions payload."""

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from natasha import AddrExtractor, Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter

from .contracts import (
    TOOL_ARGS, AgentError, PrivateRequest, PrivacyEscalation, RuntimeData, SafeRequest, tool_schema,
)


VERSION = "ru-local-v1"
EMAIL = re.compile(
    r"(?<![\w.+-])[\w.+%-]+(?:\s*[.]\s*[\w+%-]+)*\s*(?:@|[\[({<⟦](?:at|собака)[\])}>⟧])\s*"
    r"[\w-]+(?:(?:\.(?!\s)|[ \t]+\.[ \t]*|"
    r"\.[ \t]+(?=(?:ru|рф|com|net|org|info|example|test|biz|io|co)\b)|"
    r"[ \t]*[\[({<⟦](?:dot|точка)[\])}>⟧][ \t]*)[\w-]+)+", re.I,
)
PHONE = re.compile(
    r"(?<![\w\d])(?:\+\s*\d{1,3}[\s().-]*)?(?:[78][\s().-]*)?"
    r"\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?![\w\d])",
)
PAYMENT = re.compile(r"(?<![\w\d])\d(?:[ \t-]*\d){12,19}(?![\w\d])")
MASKED_CARD = re.compile(r"(?:[*•xX]{2,}\s*)+\d{4}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30}\b")
FOUR_DIGITS = re.compile(
    r"\b(?:последн\w*\s+(?:четыре|4)?\s*цифр\w*|оканчива\w*\s+на|окончани[ея]\w*)"
    r"\s*[:№-]?\s*(\d{4})\b", re.I,
)
ADDRESS_UNIT = re.compile(
    r"(?<!\w)(?:д(?:ом)?|корп(?:ус)?|кв(?:артира)?|строение|подъезд|офис|индекс)"
    r"\.?\s*(?:№\s*)?\d{1,6}(?:[/-][\dА-Яа-яA-Za-z]+|[А-Яа-яA-Za-z](?!\w))?\b", re.I,
)
# Treat an explicitly labelled address as a single span. Removing only the
# post code or street prefix could otherwise leave an unknown street visible.
ADDRESS_CUE = re.compile(
    r"\b(?:по\s+адресу\s*[:,-]?\s*|адрес(?:\s+(?:доставки|проживания|получателя))?\s*[:=]\s*)"
    r"([^\s\[;!?][^;!?\n]*?)(?=[;!?\n]|\.\s+(?=(?:верн|прошу|товар|заказ|не\s|мне\s|"
    r"хочу|нуж|добрый|спасибо|возвр|достав))|$)", re.I,
)
# Stop if a contact introducer survives all recognizers with an unparsed value.
RESIDUAL = re.compile(
    r"@|[\[({<⟦](?:at|собака|dot|точка)[\])}>⟧]|(?<!\w)(?:https?://|www\.)"
    r"|\b(?:телефон|мобильный|реквизиты|адрес|ФИО)\s*:\s*(?!\s*\[)[\w+]"
    r"|(?<!\w)\+(?:[\s().-]*\d){7,}"
    r"|(?<!\w)(?:ул\.|пер\.|пр-т|д\.|кв\.)\s*(?!\s*\[)[\w]", re.I,
)
BUSINESS_WORDS = re.compile(
    r"\b(?:не|нет|без|брак\w*|дефект\w*|неисправ\w*|слом\w*|работа\w*|включа\w*|"
    r"возврат\w*|верну\w*|верни\w*|возвра\w*|переве\w*|перечисл\w*|компенсаци\w*|"
    r"друг(?:ая|ую|ие|ой)|прежн\w*|исходн\w*|\d+\s*(?:шт\w*|единиц\w*))\b", re.I,
)
NAME_CUE = re.compile(
    r"\b(?:меня зовут|мо[её] имя|ФИО|получатель|контактное лицо|с уважением)[,: ]+\s*"
    r"([А-ЯЁA-Z][а-яёa-z-]+(?:[ \t]+[А-ЯЁA-Z][а-яёa-z-]+){0,2})", re.I,
)
TRANSFER = re.compile(
    r"\b(?:перев(?:еди(?:те)?|ест[ьи])|перечисл(?:и(?:те)?|ить)|верн(?:и(?:те)?|уть)"
    r"|оформ(?:и(?:те)?|ить)\s+возврат|возврат(?:ить)?)\b", re.I,
)
DIRECTED_CARD = re.compile(
    r"\bна\s+(?:(?:мою|банковскую|следующую|другую|новую|эту|вот\s+эту)\s+)*"
    r"карт[уы]\s*(?:(?:с\s+номером|номер|№)\s*)?[:№-]?\s*$", re.I,
)
GENERIC_ADDRESS_WORDS = {
    "улица", "улицы", "переулок", "проспект", "город", "дом", "корпус", "квартира", "область",
    "площадь", "набережная", "проезд", "бульвар", "строение", "квартал", "россия", "шоссе",
}
_BODY_KEYS = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "temperature", "max_tokens"}
_TOOL_FIELDS = {
    "sanitized_text", "policy", "as_of", "auto_refund_limit", "currency", "payment_details_supplied",
    "destination_is_definitely_different", "orders", "id", "title", "text", "order_id", "ordered_at",
    "delivered_at", "lines", "sku", "name", "category", "qty", "remaining_qty", "paid_unit_price",
    "returnable", "warranty_months", "decision", "policy_rules", "reason_code", "proposed_refund_total",
    "refund", "amount", "total", "intent", "items", "reason", "evidence", "alternative_payment_requested",
    "uncertainties", "code", "error",
}
_PROTOCOL_KEYS = {
    "system": {"role", "content"}, "user": {"role", "content"},
    "assistant": {"role", "content", "tool_calls"}, "tool": {"role", "content", "tool_call_id", "name"},
}


def _blocked():
    raise AgentError("PRIVACY_BLOCKED")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    text = text.translate(str.maketrans("‐‑‒–—−", "------"))
    return re.sub(r"[^\S\n]+", " ", text).strip()


def _terms_pattern(terms):
    expressions = set()
    for term in terms:
        words = re.findall(r"[^\W_]+", _normalize(term).casefold().replace("ё", "е"))
        if words:
            parts = [re.escape(w).replace("е", "[её]") for w in words]
            expressions.add(r"[\W_]+".join(parts))
    if not expressions:
        return re.compile(r"(?!)")
    return re.compile(r"(?<!\w)(?:" + "|".join(sorted(expressions, key=len, reverse=True)) + r")(?!\w)", re.I)


def _overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


class PrivacyGate:
    def __init__(self, data: RuntimeData):
        self._by_email = {c.email.strip().casefold(): c.customer_id for c in data.customers}
        self._orders = data.orders
        self._rules = {p.id: p.model_dump() for p in data.policy}
        self._schemas = {name: tool_schema(name) for name in TOOL_ARGS}
        self._active = None
        self._ready = False
        names, addresses, contacts = set(), set(), set()
        for c in data.customers:
            names.add(c.full_name)
            names.update(re.findall(r"[^\W\d_]+", c.full_name))
            addresses.add(c.address)
            for word in re.findall(r"[^\W\d_]{3,}", c.address):
                if word.casefold() not in GENERIC_ADDRESS_WORDS:
                    addresses.add(word)
            contacts.add(c.email)
        self._phones = []
        for c in data.customers:
            digits = re.sub(r"\D", "", c.phone)
            if len(digits) >= 7:
                if len(digits) == 11 and digits[0] in "78":
                    prefix, tail = r"(?:\+?[78][\s().-]*)?", digits[1:]
                else:
                    prefix, tail = r"\+?", digits
                self._phones.append(re.compile(r"(?<!\w)" + prefix + r"[\s().-]*".join(tail) + r"(?!\w)"))
        trusted = [p.name for p in data.catalog] + [p.sku for p in data.catalog]
        trusted.extend(o.order_id for o in data.orders)
        # Product brand/model spans can look like people to a Russian news NER model.
        for p in data.catalog:
            trusted.extend(re.findall(r"[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9][A-Za-z0-9-]*)*", p.name))
        self._products = _terms_pattern(trusted)
        try:
            self._segment = Segmenter()
            self._ner = NewsNERTagger(NewsEmbedding())
            morph = MorphVocab()
            self._addr = AddrExtractor(morph)
            for word in tuple(names):
                if word.isalpha():
                    for parse in morph.parse(word):
                        if {"Name", "Patr", "Surn"} & parse.tag.grammemes:
                            names.update(form.word for form in parse.lexeme)
            self._ready = True
        except Exception:
            # Model/dictionary loading must fail closed. Unknown sender remains a local P8.
            pass
        self._known = [("EMAIL", _terms_pattern(contacts)), ("PERSON", _terms_pattern(names)),
                       ("ADDRESS", _terms_pattern(addresses))]
        self._nlp = lru_cache(maxsize=256)(self._nlp_spans)

    def _nlp_spans(self, text):
        shielded = self._products.sub(lambda m: " " * len(m[0]), text)
        if not re.search(r"[^\W\d_]", shielded):
            return ()
        doc = Doc(shielded)
        doc.segment(self._segment)
        if doc.tokens:
            doc.tag_ner(self._ner)
        spans = [(s.start, s.stop, {"PER": "PERSON", "LOC": "ADDRESS"}[s.type])
                 for s in doc.spans or () if s.type in {"PER", "LOC"}]
        spans.extend((m.start, m.stop, "ADDRESS") for m in self._addr(shielded))
        return tuple(spans)

    def _spans(self, text):
        patterns = [
            ("PAYMENT_DETAILS", PAYMENT), ("PAYMENT_DETAILS", MASKED_CARD), ("PAYMENT_DETAILS", IBAN),
            ("EMAIL", EMAIL), ("PHONE", PHONE), *(("PHONE", p) for p in self._phones),
            *self._known, ("ADDRESS", ADDRESS_UNIT),
        ]
        spans = [(m.start(), m.end(), kind) for kind, pattern in patterns for m in pattern.finditer(text)]
        spans.extend((*m.span(1), "PAYMENT_DETAILS") for m in FOUR_DIGITS.finditer(text))
        spans.extend((*m.span(1), "PERSON") for m in NAME_CUE.finditer(text))
        spans.extend((*m.span(1), "ADDRESS") for m in ADDRESS_CUE.finditer(text))
        spans.extend(self._nlp(text))
        return spans

    def _payment_mismatch(self, text, customer_id):
        orders = [o for o in self._orders if o.customer_id == customer_id]
        explicit = [o for o in orders if re.search(r"(?<!\w)" + re.escape(o.order_id) + r"(?!\w)", text, re.I)]
        candidates = explicit or orders
        for match in PAYMENT.finditer(text):
            digits = re.sub(r"\D", "", match[0])
            if not (13 <= len(digits) <= 19) or not candidates:
                continue
            # Look only at an explicit destination in the same clause. A number
            # in a signature, a quotation without a transfer, or a negated
            # transfer has no authority over the original payment method.
            clause = re.split(r"[.!?;\n]", text[:match.start()])[-1]
            verbs = list(TRANSFER.finditer(clause))
            if not DIRECTED_CARD.search(clause) or not verbs:
                continue
            verb = verbs[-1]
            if (re.search(r"\bне(?:\s+(?:надо|нужно|прошу|стоит|следует))?\s*$", clause[:verb.start()], re.I)
                    or re.search(r"\bне\b|\bбез\b", clause[verb.end():], re.I)):
                continue
            if all(not o.payment_method.endswith(" " + digits[-4:]) for o in candidates):
                return True
        return None

    def prepare(self, request: PrivateRequest) -> SafeRequest | PrivacyEscalation:
        self._active = None
        customer_id = self._by_email.get(request.from_email.strip().casefold())
        if customer_id is None:
            return PrivacyEscalation(code="UNKNOWN_SENDER")
        try:
            if not self._ready:
                _blocked()
            raw = _normalize(request.text)
            spans = sorted(self._spans(raw), key=lambda s: (s[0], -s[1], s[2]))
            protected = [m.span() for m in self._products.finditer(raw)]
            if any(_overlaps(span, term) for span in spans for term in protected):
                _blocked()
            merged = []
            for start, end, kind in spans:
                if merged and start < merged[-1][1]:
                    prev_start, prev_end, prev_kind = merged[-1]
                    # Contact patterns may overlap longer payment or address spans.
                    merged[-1] = (prev_start, max(end, prev_end), prev_kind)
                else:
                    merged.append((start, end, kind))
            parts, counts, cursor = [], Counter(), 0
            for start, end, kind in merged:
                parts.extend([raw[cursor:start], f"[{kind}]"])
                counts[kind] += 1
                cursor = end
            sanitized = "".join([*parts, raw[cursor:]])
            remaining = re.sub(r"\[(?:PERSON|ADDRESS|EMAIL|PHONE|PAYMENT_DETAILS)\]", " ", sanitized)
            if not re.search(r"[^\W\d_]", remaining) or RESIDUAL.search(sanitized):
                _blocked()
            if BUSINESS_WORDS.findall(raw) != BUSINESS_WORDS.findall(sanitized):
                _blocked()
            self._inspect_string(sanitized)
            safe = SafeRequest(
                request_id=request.request_id, customer_id=customer_id, sanitized_text=sanitized,
                privacy_version=VERSION, redactions=dict(counts),
                payment_details_supplied=any(s[2] == "PAYMENT_DETAILS" for s in spans),
                destination_is_definitely_different=self._payment_mismatch(raw, customer_id),
            )
            self._active = safe
            return safe
        except Exception:
            return PrivacyEscalation(code="PRIVACY_BLOCKED")

    def _inspect_string(self, value):
        text = _normalize(value)
        if self._spans(text) or RESIDUAL.search(text):
            _blocked()

    def _inspect_tree(self, value):
        if isinstance(value, dict):
            if value.keys() - _TOOL_FIELDS:
                _blocked()
            if ("text" in value or "title" in value) and value != self._rules.get(value.get("id")):
                _blocked()
            if "sanitized_text" in value:
                if self._active is None or value["sanitized_text"] != self._active.sanitized_text:
                    _blocked()
                owned = {o.order_id for o in self._orders if o.customer_id == self._active.customer_id}
                if any(order["order_id"] not in owned for order in value.get("orders", [])):
                    _blocked()
            for child in value.values():
                self._inspect_tree(child)
        elif isinstance(value, list):
            for child in value:
                self._inspect_tree(child)
        elif isinstance(value, str):
            # Tool content is JSON in a string. Walk decoded values so escaped
            # Unicode and nested keys receive the same inspection as plain text.
            candidate = value.lstrip()
            if candidate.startswith(("{", "[")):
                try:
                    decoded = json.loads(candidate)
                except ValueError:
                    pass   # a redaction marker at the start is ordinary prose
                else:
                    self._inspect_tree(decoded)
                    return
            self._inspect_string(value)
        elif value is not None:
            if not isinstance(value, (int, bool, float)):
                _blocked()
            self._inspect_string(str(value))

    def _check_message(self, message):
        role = message["role"]
        if role not in _PROTOCOL_KEYS or message.keys() - _PROTOCOL_KEYS[role]:
            _blocked()
        if "content" in message and message["content"] is not None:
            if not isinstance(message["content"], str):
                _blocked()
            self._inspect_tree(message["content"])
        if role == "tool":
            self._inspect_string(message["tool_call_id"])
            if "name" in message and message["name"] not in TOOL_ARGS:
                _blocked()
        for call in message.get("tool_calls", []):
            if set(call) != {"id", "type", "function"} or call["type"] != "function":
                _blocked()
            self._inspect_string(call["id"])
            func = call["function"]
            if set(func) != {"name", "arguments"} or func["name"] not in TOOL_ARGS:
                _blocked()
            args = json.loads(func["arguments"])
            TOOL_ARGS[func["name"]].model_validate(args)
            self._inspect_tree(args)

    def guard(self, payload: dict) -> None:
        """Fail closed on the complete final wire object; no headers or exceptions are logged."""
        try:
            if not self._ready or payload.keys() - _BODY_KEYS:
                _blocked()
            self._inspect_string(payload["model"])
            for schema in payload["tools"]:
                if schema != self._schemas[schema["function"]["name"]]:
                    _blocked()
            choice = payload["tool_choice"]
            if isinstance(choice, dict):
                if choice != {"type": "function", "function": {"name": choice["function"]["name"]}}:
                    _blocked()
                if choice["function"]["name"] not in TOOL_ARGS:
                    _blocked()
            elif choice not in {"auto", "required", "none"}:
                _blocked()
            for key in ("parallel_tool_calls", "temperature", "max_tokens"):
                if key in payload:
                    if type(payload[key]) not in (bool, int, float):
                        _blocked()
                    self._inspect_tree(payload[key])
            for message in payload["messages"]:
                self._check_message(message)
        except Exception:
            raise AgentError("PRIVACY_BLOCKED") from None
