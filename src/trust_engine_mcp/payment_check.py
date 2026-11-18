"""check_payment: ask AgentTrustAPI "may this agent make exactly this payment?" before the agent pays.

UNRELEASED (pending owner decision 22.11). Contract, from the agent-trust-api repository:

- ``POST https://api.agenttrustapi.com/v1/authority/payment/match`` (docs/authority-http-v1.md,
  docs/production-mainnet.md). Body: ``payment-authority-match-input/v1`` = ``{version, evidence, request}``
  (docs/authority-wire-v1.md). ``evidence`` is already-normalized ``authority-evidence/v1`` produced upstream (it
  carries an ``evidence_id`` that only the frozen ATA builder can compute, so this tool never builds or edits it).
  ``request`` is ``payment-authority-request/v1`` = ``{version, resource, amount, asset, network,
  additional_constraints, bindings}``.
- ``200``: ``payment-authority-profile/v1`` = ``{version, matched, profile_valid, profile_reason_codes,
  evidence_id, matcher_ran, matcher_version, matcher_reason_codes, matcher_matched, verification_usable,
  verification_reason_codes}`` (src/agent_trust_api/payment_authority_profile.py). A mismatch is HTTP 200 with
  ``matched: false``.
- ``402``: x402 v2 ``{x402Version, error, resource, accepts[{scheme, network, asset, amount, payTo,
  maxTimeoutSeconds, extra}], extensions}``, also base64 JSON in the ``PAYMENT-REQUIRED`` header. Production price
  ``10000`` atomic USDC (0.01 USDC) on Base mainnet ``eip155:8453``. A paid retry sends ``PAYMENT-SIGNATURE``
  (src/agent_trust_api/payment.py). ``409``: that payment identity was already used.
- ``422 authority_wire_invalid``, ``413``, ``415``: ``{error, detail}``. The x402 middleware settles only responses
  below 400, so these are not charged.

The endpoint itself does **not** emit ALLOW / REVIEW / BLOCK. The verdict here is this tool's projection, using
the authority rule of ATA-016 (docs/ata-016-payment-preflight.md, src/agent_trust_api/preflight.py):
matched -> ALLOW (authority only); invalid, mismatched or not matched authority -> BLOCK; check missing or
incomplete (payment required, service unreachable, unexpected answer) -> REVIEW.

The tool holds no key and never pays. On 402 it returns the payment requirements so the calling agent's own wallet
can pay, then the agent calls again with ``payment_signature``. The signature is sent only as the
``PAYMENT-SIGNATURE`` header and is never returned, logged or stored.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from collections.abc import Callable
from typing import Any

import httpx

from trust_engine_mcp.data import OFFLINE_ENV, TIMEOUT_SECONDS, user_agent

API_BASE_URL_ENV = "TRUST_ENGINE_API_BASE_URL"
DEFAULT_API_BASE_URL = "https://api.agenttrustapi.com"
MATCH_PATH = "/v1/authority/payment/match"

WIRE_VERSION = "payment-authority-match-input/v1"
REQUEST_VERSION = "payment-authority-request/v1"
EVIDENCE_VERSION = "authority-evidence/v1"
RESULT_VERSION = "payment-authority-profile/v1"

HEADER_PAYMENT_SIGNATURE = "PAYMENT-SIGNATURE"
HEADER_PAYMENT_REQUIRED = "PAYMENT-REQUIRED"
HEADER_PAYMENT_RESPONSE = "PAYMENT-RESPONSE"

MAX_BODY_BYTES = 256 * 1024  # the endpoint's own limit
MAX_SIGNATURE_CHARS = 16 * 1024
MAX_FIELD_CHARS = 2048
MAX_LIST_ITEMS = 64

_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", re.ASCII)  # ATA-009D exact decimal
_BASE64_RE = re.compile(r"[A-Za-z0-9+/_=-]+", re.ASCII)

VERDICT_SOURCE = (
    "trust-engine-mcp projection of the AgentTrustAPI payment-authority-profile/v1 answer, using the ATA-016 "
    "authority rule (matched -> ALLOW; not matched or invalid -> BLOCK; no complete answer -> REVIEW). "
    "The endpoint itself returns matched true/false with reason codes."
)
SCOPE_NOTE = (
    "Checks one thing: does the supplied, already-normalized delegated authority cover exactly this payment "
    "(resource, amount, asset, network, delegation depth, bindings, validity, revocation). It does not verify raw "
    "credentials, the x402 challenge of the payee, the payee's identity, the asset's safety, or the payment itself. "
    "It gives no legal guarantee and no insurance."
)

ClientFactory = Callable[[str], httpx.Client]


class PaymentCheckInputError(ValueError):
    """The arguments cannot form a valid request; the message is safe to show to the agent."""


def api_base_url(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return (env.get(API_BASE_URL_ENV) or DEFAULT_API_BASE_URL).rstrip("/")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise PaymentCheckInputError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise PaymentCheckInputError(f"{name} must not be empty")
    if value != value.strip():
        raise PaymentCheckInputError(f"{name} must not start or end with spaces (matching is exact)")
    if len(value) > MAX_FIELD_CHARS:
        raise PaymentCheckInputError(f"{name} is longer than {MAX_FIELD_CHARS} characters")
    return value


def _facts(items: Any, name: str, keys: tuple[str, ...]) -> list[dict[str, str]]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise PaymentCheckInputError(f"{name} must be a list of objects with keys {', '.join(keys)}")
    if len(items) > MAX_LIST_ITEMS:
        raise PaymentCheckInputError(f"{name} has more than {MAX_LIST_ITEMS} entries")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != set(keys):
            raise PaymentCheckInputError(f"{name}[{i}] must be an object with exactly the keys {', '.join(keys)}")
        out.append({k: _string(item[k], f"{name}[{i}].{k}") for k in keys})
    return out


def build_match_input(
    evidence: Any,
    resource: Any,
    amount: Any,
    asset: Any,
    network: Any,
    bindings: Any = None,
    additional_constraints: Any = None,
) -> dict[str, Any]:
    """The exact ``payment-authority-match-input/v1`` body. Raises PaymentCheckInputError with a clear message."""
    if not isinstance(evidence, dict) or not evidence:
        raise PaymentCheckInputError(
            "evidence must be the normalized authority-evidence/v1 object (with evidence_id) from your authority "
            "adapter; this tool does not build or change it"
        )
    if evidence.get("version") != EVIDENCE_VERSION:
        raise PaymentCheckInputError(f'evidence.version must be "{EVIDENCE_VERSION}"')
    if isinstance(amount, (bool, int, float)):
        raise PaymentCheckInputError('amount must be a decimal string such as "10.00", not a number')
    amount_s = _string(amount, "amount")
    if _DECIMAL_RE.fullmatch(amount_s) is None:
        raise PaymentCheckInputError('amount must be a plain decimal string such as "10.00" (no sign, exponent or +)')
    constraints = _facts(additional_constraints, "additional_constraints", ("name", "value"))
    if any(c["name"] == "delegation_depth" for c in constraints):
        raise PaymentCheckInputError(
            "additional_constraints must not contain delegation_depth: AgentTrustAPI derives it from the delegation "
            "chain"
        )
    return {
        "version": WIRE_VERSION,
        "evidence": evidence,
        "request": {
            "version": REQUEST_VERSION,
            "resource": _string(resource, "resource"),
            "amount": amount_s,
            "asset": _string(asset, "asset"),
            "network": _string(network, "network"),
            "additional_constraints": constraints,
            "bindings": _facts(bindings, "bindings", ("kind", "value")),
        },
    }


def _check_signature(payment_signature: Any) -> str | None:
    if payment_signature is None:
        return None
    if not isinstance(payment_signature, str) or not payment_signature.strip():
        raise PaymentCheckInputError("payment_signature must be the base64 x402 PAYMENT-SIGNATURE value")
    sig = payment_signature.strip()
    if len(sig) > MAX_SIGNATURE_CHARS or _BASE64_RE.fullmatch(sig) is None:
        # Never echo the value back.
        raise PaymentCheckInputError("payment_signature must be the base64 x402 PAYMENT-SIGNATURE value")
    return sig


def _decode_b64_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(padded) if ("-" in value or "_" in value) else base64.b64decode(padded)
        data = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    try:
        data = response.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _requirements(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    accepts = (body or {}).get("accepts")
    out = []
    for a in accepts if isinstance(accepts, list) else []:
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "scheme": a.get("scheme"),
                "network": a.get("network"),
                "asset": a.get("asset"),
                "amount": a.get("amount"),
                "pay_to": a.get("payTo"),
                "max_timeout_seconds": a.get("maxTimeoutSeconds"),
                "extra": a.get("extra") if isinstance(a.get("extra"), dict) else {},
            }
        )
    return out


def _valid_result(data: dict[str, Any] | None) -> bool:
    if not data or data.get("version") != RESULT_VERSION or not isinstance(data.get("matched"), bool):
        return False
    for key in ("profile_reason_codes", "matcher_reason_codes", "verification_reason_codes"):
        codes = data.get(key)
        if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
            return False
    return True


def _answer(verdict: str, status: str, reasons: list[str], **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "verdict": verdict,
        "status": status,
        "reasons": reasons,
        "pay_now": verdict == "ALLOW",
        "verdict_source": VERDICT_SOURCE,
        "scope": SCOPE_NOTE,
    }
    out.update(extra)
    return out


class PaymentChecker:
    def __init__(
        self,
        channel: str = "mcp",
        base_url: str | None = None,
        offline: bool | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.channel = channel
        self.base_url = (base_url or api_base_url()).rstrip("/")
        self.offline = (
            (os.environ.get(OFFLINE_ENV) or "").strip() in ("1", "true", "yes") if offline is None else offline
        )
        self._client_factory = client_factory or (
            lambda base: httpx.Client(base_url=base, timeout=TIMEOUT_SECONDS, follow_redirects=False)
        )

    @property
    def endpoint(self) -> str:
        return self.base_url + MATCH_PATH

    def check_payment(
        self,
        evidence: Any,
        resource: Any,
        amount: Any,
        asset: Any,
        network: Any,
        bindings: Any = None,
        additional_constraints: Any = None,
        agent_id: Any = None,
        payment_signature: Any = None,
    ) -> dict[str, Any]:
        body = build_match_input(evidence, resource, amount, asset, network, bindings, additional_constraints)
        sig = _check_signature(payment_signature)
        payment = {k: body["request"][k] for k in ("resource", "amount", "asset", "network")}
        base = {"endpoint": self.endpoint, "payment": payment}

        if agent_id is not None:
            agent = _string(agent_id, "agent_id")
            subject = evidence.get("subject_agent")
            subject_id = subject.get("identifier") if isinstance(subject, dict) else None
            if agent != subject_id:
                # ATA-016: agent identity that disagrees with the authority subject is explicit negative evidence.
                return _answer(
                    "BLOCK",
                    "checked_locally",
                    ["AGENT_ID_NOT_AUTHORITY_SUBJECT"],
                    detail="agent_id differs from evidence.subject_agent.identifier; no paid call was made.",
                    **base,
                )

        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_BODY_BYTES:
            raise PaymentCheckInputError(f"request body is larger than {MAX_BODY_BYTES // 1024} KiB")

        if self.offline:
            return _answer(
                "REVIEW",
                "not_checked",
                ["CHECK_NOT_RUN_OFFLINE"],
                detail=f"{OFFLINE_ENV}=1: no request was sent. Do not treat this as permission to pay.",
                **base,
            )

        headers = {
            "User-Agent": user_agent(self.channel),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if sig:
            headers[HEADER_PAYMENT_SIGNATURE] = sig
        try:
            with self._client_factory(self.base_url) as client:
                r = client.post(MATCH_PATH, content=encoded, headers=headers)
        except httpx.TimeoutException:
            return _answer(
                "REVIEW",
                "not_checked",
                ["CHECK_UNAVAILABLE_TIMEOUT"],
                detail=f"No answer within {TIMEOUT_SECONDS:.0f} s. Do not pay on this basis; try again later.",
                **base,
            )
        except httpx.HTTPError as exc:
            return _answer(
                "REVIEW",
                "not_checked",
                ["CHECK_UNAVAILABLE_NETWORK"],
                detail=f"{type(exc).__name__}: the service could not be reached. Do not pay on this basis.",
                **base,
            )
        return self._interpret(r, sig is not None, base)

    def _interpret(self, r: httpx.Response, signed: bool, base: dict[str, Any]) -> dict[str, Any]:
        data = _json_object(r)
        if r.status_code == 200:
            if not _valid_result(data):
                return _answer(
                    "REVIEW",
                    "not_checked",
                    ["CHECK_RESPONSE_MALFORMED"],
                    detail="HTTP 200 but not a payment-authority-profile/v1 answer. Do not pay on this basis.",
                    **base,
                )
            assert data is not None
            reasons: list[str] = []
            for key in ("profile_reason_codes", "matcher_reason_codes", "verification_reason_codes"):
                for code in data[key]:
                    if code not in reasons:
                        reasons.append(code)
            out = _answer(
                "ALLOW" if data["matched"] else "BLOCK",
                "checked",
                ["AUTHORITY_MATCHED" if data["matched"] else "AUTHORITY_NOT_MATCHED", *reasons],
                evidence_id=data.get("evidence_id"),
                api_result=data,
                **base,
            )
            settlement = _decode_b64_json(r.headers.get(HEADER_PAYMENT_RESPONSE))
            if settlement:
                out["check_payment_settlement"] = {
                    k: settlement.get(k) for k in ("success", "transaction", "network", "payer")
                }
            return out
        if r.status_code == 402:
            body = data if data and "accepts" in data else _decode_b64_json(r.headers.get(HEADER_PAYMENT_REQUIRED))
            reqs = _requirements(body)
            reason = "PAYMENT_SIGNATURE_REJECTED" if signed else "PAYMENT_REQUIRED_FOR_CHECK"
            return _answer(
                "REVIEW",
                "payment_required",
                [reason],
                payment_required={
                    "x402_version": (body or {}).get("x402Version"),
                    "error": (body or {}).get("error"),
                    "accepts": reqs,
                    "resource": ((body or {}).get("resource") or {}).get("url")
                    if isinstance((body or {}).get("resource"), dict)
                    else None,
                },
                how_to_pay=(
                    "This check costs the amount in accepts (production: 10000 atomic USDC = 0.01 USDC on Base, "
                    "eip155:8453). This tool holds no key and pays nothing. If your owner allows it, sign an x402 v2 "
                    "'exact' payment for one entry of accepts with your own wallet, then call check_payment again "
                    "with the same arguments and payment_signature=<base64 PAYMENT-SIGNATURE value>. "
                    "Until then, do not make the payment you wanted to check."
                ),
                **base,
            )
        if r.status_code == 422 and data and data.get("error") == "authority_wire_invalid":
            return _answer(
                "BLOCK",
                "checked",
                ["AUTHORITY_EVIDENCE_INVALID"],
                detail=str(data.get("detail") or "")[:300],
                **base,
            )
        if r.status_code == 409:
            return _answer(
                "REVIEW",
                "payment_required",
                ["PAYMENT_SIGNATURE_ALREADY_USED"],
                detail="That payment signature was already used for another request; sign a new payment.",
                **base,
            )
        error = data.get("error") if data else None
        return _answer(
            "REVIEW",
            "not_checked",
            [f"CHECK_FAILED_HTTP_{r.status_code}"],
            detail=(str(error)[:300] if error else f"HTTP {r.status_code}") + ". Do not pay on this basis.",
            **base,
        )
