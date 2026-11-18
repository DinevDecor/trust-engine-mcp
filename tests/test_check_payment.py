"""check_payment (unreleased): a fake api.agenttrustapi.com (httpx.MockTransport). No network, no payments."""

import asyncio
import base64
import json

import httpx
import pytest

from trust_engine_mcp import data as dm
from trust_engine_mcp import payment_check as pc
from trust_engine_mcp import server as ms
from trust_engine_mcp.tools import ToolFailure, TrustEngineTools

# The canonical example from agent-trust-api docs/authority-wire-v1.md (non-secret example values).
EVIDENCE = {
    "version": "authority-evidence/v1",
    "kind": "delegated-payment-authority",
    "evaluated_at": "2026-09-07T10:00:00.000000Z",
    "verifier": {
        "identifier": "agent-trust-api",
        "identity_type": "service",
        "verification_method": "normalized-adapter:v1",
    },
    "principal": {"identifier": "did:key:zExampleMerchant", "identity_type": "did", "verification_method": "did:key"},
    "subject_agent": {"identifier": "did:key:zExampleAgent", "identity_type": "did", "verification_method": "did:key"},
    "action": "payment.initiate",
    "resource": "merchant:example-store",
    "constraints": [
        {"name": "amount", "operator": "lte", "value": "25.00"},
        {"name": "asset", "operator": "eq", "value": "USDC"},
        {"name": "delegation_depth", "operator": "lte", "value": "1"},
        {"name": "network", "operator": "eq", "value": "eip155:84532"},
    ],
    "validity": {"not_before": "2026-09-07T09:00:00.000000Z", "expires_at": "2026-09-07T11:00:00.000000Z"},
    "delegation_chain": [],
    "revocation": {
        "status": "VALID",
        "source": "https://issuer.example/status/example",
        "checked_at": "2026-09-07T09:55:00.000000Z",
        "fresh_until": "2026-09-07T10:30:00.000000Z",
        "source_artifact_hash": "b" * 64,
    },
    "bindings": [{"kind": "audience", "value": "merchant:example-store"}],
    "source": {
        "source_id": "example-credential",
        "source_format": "example-format",
        "adapter_version": "adapter-1",
        "artifact_hash": "a" * 64,
    },
    "evidence_id": "a6710b4db5391ca734845f9de2cf31d89d8ddfd6fa9498581a8efc4c72a0c197",
}
ARGS = {
    "evidence": EVIDENCE,
    "resource": "merchant:example-store",
    "amount": "10.00",
    "asset": "USDC",
    "network": "eip155:84532",
    "bindings": [{"kind": "audience", "value": "merchant:example-store"}],
}
SECRET_SIG = base64.b64encode(json.dumps({"x402Version": 2, "payload": {"signature": "0xSECRET"}}).encode()).decode()


def profile(matched, profile_codes=("PAYMENT_AUTHORITY_PROFILE_VALID",), matcher_codes=("EVIDENCE_MATCHED",)):
    return {
        "version": "payment-authority-profile/v1",
        "matched": matched,
        "profile_valid": True,
        "profile_reason_codes": list(profile_codes),
        "evidence_id": EVIDENCE["evidence_id"],
        "matcher_ran": True,
        "matcher_version": "authority-matcher/v1",
        "matcher_reason_codes": list(matcher_codes),
        "matcher_matched": matched,
        "verification_usable": True,
        "verification_reason_codes": ["EVIDENCE_USABLE"],
    }


PAYMENT_REQUIRED = {
    "x402Version": 2,
    "error": "PAYMENT-SIGNATURE header is required",
    "resource": {"url": "https://api.agenttrustapi.com/v1/authority/payment/match", "mimeType": "application/json"},
    "accepts": [
        {
            "scheme": "exact",
            "network": "eip155:8453",
            "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "amount": "10000",
            "payTo": "0x8b0ad0CE12Ac86822c4d5dE442F5e445d14F9183",
            "maxTimeoutSeconds": 60,
            "extra": {"name": "USD Coin", "version": "2"},
        }
    ],
    "extensions": {"bazaar": {}},
}


class FakeApi:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.handler(request)


def checker(handler, offline=False):
    api = FakeApi(handler)
    c = pc.PaymentChecker(
        channel="mcp",
        base_url="https://api.example",
        offline=offline,
        client_factory=lambda base: httpx.Client(base_url=base, transport=httpx.MockTransport(api)),
    )
    return c, api


def ok(body, status=200, headers=None):
    return lambda request: httpx.Response(status, json=body, headers=headers)


# --- verdicts ------------------------------------------------------------------------------------------------------
def test_allow_when_authority_matches_and_request_is_exact_contract():
    c, api = checker(ok(profile(True)))
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "ALLOW" and r["pay_now"] is True and r["status"] == "checked"
    assert r["reasons"][0] == "AUTHORITY_MATCHED" and "EVIDENCE_MATCHED" in r["reasons"]
    assert r["api_result"]["matched"] is True and r["evidence_id"] == EVIDENCE["evidence_id"]
    (req,) = api.requests
    assert req.method == "POST" and req.url.path == "/v1/authority/payment/match"
    assert req.headers["content-type"] == "application/json"
    assert req.headers["user-agent"] == f"trust-engine-mcp/{dm.__version__} (mcp)"
    assert "payment-signature" not in req.headers
    sent = json.loads(req.content)
    assert set(sent) == {"version", "evidence", "request"} and sent["version"] == "payment-authority-match-input/v1"
    assert sent["evidence"] == EVIDENCE
    assert sent["request"] == {
        "version": "payment-authority-request/v1",
        "resource": "merchant:example-store",
        "amount": "10.00",
        "asset": "USDC",
        "network": "eip155:84532",
        "additional_constraints": [],
        "bindings": [{"kind": "audience", "value": "merchant:example-store"}],
    }


def test_block_when_authority_does_not_match():
    body = profile(False, matcher_codes=("REQUEST_CONSTRAINT_LIMIT_EXCEEDED",))
    c, _ = checker(ok(body))
    r = c.check_payment(**dict(ARGS, amount="30.00"))
    assert r["verdict"] == "BLOCK" and r["pay_now"] is False
    assert r["reasons"][0] == "AUTHORITY_NOT_MATCHED" and "REQUEST_CONSTRAINT_LIMIT_EXCEEDED" in r["reasons"]


def test_block_when_evidence_wire_invalid_422():
    c, _ = checker(ok({"error": "authority_wire_invalid", "detail": "evidence.evidence_id mismatch"}, status=422))
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "BLOCK" and r["reasons"] == ["AUTHORITY_EVIDENCE_INVALID"]
    assert "evidence_id mismatch" in r["detail"]


def test_block_locally_when_agent_is_not_the_authority_subject():
    c, api = checker(ok(profile(True)))
    r = c.check_payment(**ARGS, agent_id="did:key:zSomeoneElse")
    assert r["verdict"] == "BLOCK" and r["reasons"] == ["AGENT_ID_NOT_AUTHORITY_SUBJECT"]
    assert api.requests == []  # no paid call
    assert c.check_payment(**ARGS, agent_id="did:key:zExampleAgent")["verdict"] == "ALLOW"


def test_review_when_payment_required_402_with_x402_requirements():
    c, _ = checker(ok(PAYMENT_REQUIRED, status=402))
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "REVIEW" and r["status"] == "payment_required" and r["pay_now"] is False
    assert r["reasons"] == ["PAYMENT_REQUIRED_FOR_CHECK"]
    (req,) = r["payment_required"]["accepts"]
    assert req == {
        "scheme": "exact",
        "network": "eip155:8453",
        "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "amount": "10000",
        "pay_to": "0x8b0ad0CE12Ac86822c4d5dE442F5e445d14F9183",
        "max_timeout_seconds": 60,
        "extra": {"name": "USD Coin", "version": "2"},
    }
    assert r["payment_required"]["x402_version"] == 2
    assert "payment_signature" in r["how_to_pay"]


def test_402_requirements_read_from_header_when_body_is_empty():
    header = base64.b64encode(json.dumps(PAYMENT_REQUIRED).encode()).decode()
    c, _ = checker(lambda request: httpx.Response(402, text="", headers={"PAYMENT-REQUIRED": header}))
    r = c.check_payment(**ARGS)
    assert r["payment_required"]["accepts"][0]["pay_to"] == "0x8b0ad0CE12Ac86822c4d5dE442F5e445d14F9183"


def test_paid_retry_sends_signature_header_only_and_never_returns_it():
    settle = base64.b64encode(json.dumps({"success": True, "transaction": "0xabc", "network": "eip155:8453"}).encode())
    c, api = checker(ok(profile(True), headers={"PAYMENT-RESPONSE": settle.decode()}))
    r = c.check_payment(**ARGS, payment_signature=SECRET_SIG)
    assert api.requests[0].headers["payment-signature"] == SECRET_SIG
    assert SECRET_SIG not in json.loads(api.requests[0].content).__repr__()
    assert r["verdict"] == "ALLOW" and r["check_payment_settlement"]["transaction"] == "0xabc"
    assert SECRET_SIG not in json.dumps(r) and "0xSECRET" not in json.dumps(r)


def test_rejected_signature_and_reused_signature():
    c, _ = checker(ok(dict(PAYMENT_REQUIRED, error="payment verification failed"), status=402))
    r = c.check_payment(**ARGS, payment_signature=SECRET_SIG)
    assert (
        r["reasons"] == ["PAYMENT_SIGNATURE_REJECTED"]
        and r["payment_required"]["error"] == "payment verification failed"
    )
    assert SECRET_SIG not in json.dumps(r)
    c, _ = checker(ok({"error": "payment identity already consumed"}, status=409))
    assert c.check_payment(**ARGS, payment_signature=SECRET_SIG)["reasons"] == ["PAYMENT_SIGNATURE_ALREADY_USED"]


# --- failures are REVIEW, never ALLOW ----------------------------------------------------------------------------
def test_timeout_is_review():
    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)

    c, _ = checker(boom)
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "REVIEW" and r["reasons"] == ["CHECK_UNAVAILABLE_TIMEOUT"] and r["pay_now"] is False


def test_network_error_is_review():
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    c, _ = checker(down)
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "REVIEW" and r["reasons"] == ["CHECK_UNAVAILABLE_NETWORK"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"version": "payment-authority-profile/v1", "matched": "yes"}),
        httpx.Response(200, json=dict(profile(True), version="something-else/v9")),
        httpx.Response(200, json=dict(profile(True), matcher_reason_codes="EVIDENCE_MATCHED")),
    ],
)
def test_malformed_response_is_review(response):
    c, _ = checker(lambda request: response)
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "REVIEW" and r["reasons"] == ["CHECK_RESPONSE_MALFORMED"]


def test_server_error_is_review():
    c, _ = checker(lambda request: httpx.Response(503, json={"error": "payment settlement reconciliation required"}))
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "REVIEW" and r["reasons"] == ["CHECK_FAILED_HTTP_503"]


def test_offline_makes_no_request():
    c, api = checker(ok(profile(True)), offline=True)
    r = c.check_payment(**ARGS)
    assert r["verdict"] == "REVIEW" and r["reasons"] == ["CHECK_NOT_RUN_OFFLINE"] and api.requests == []


# --- input validation ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "change, message",
    [
        ({"evidence": None}, "evidence must be the normalized"),
        ({"evidence": dict(EVIDENCE, version="v0")}, "evidence.version"),
        ({"amount": 10}, "not a number"),
        ({"amount": "1e3"}, "plain decimal"),
        ({"amount": "-5"}, "plain decimal"),
        ({"resource": ""}, "resource must not be empty"),
        ({"asset": " USDC"}, "asset must not start or end with spaces"),
        ({"network": None}, "network must be a string"),
        ({"bindings": [{"kind": "audience"}]}, "bindings[0]"),
        ({"additional_constraints": [{"name": "delegation_depth", "value": "0"}]}, "delegation_depth"),
        ({"payment_signature": "not base64 !!"}, "payment_signature"),
    ],
)
def test_input_errors_are_clear_and_send_nothing(change, message):
    c, api = checker(ok(profile(True)))
    with pytest.raises(pc.PaymentCheckInputError, match=message.replace("[", r"\[").replace("]", r"\]")):
        c.check_payment(**dict(ARGS, **change))
    assert api.requests == []


def test_bad_signature_value_is_not_echoed():
    c, _ = checker(ok(profile(True)))
    with pytest.raises(pc.PaymentCheckInputError) as exc:
        c.check_payment(**ARGS, payment_signature="sk-live secret value!")
    assert "secret" not in str(exc.value)


def test_default_base_url_is_production_api():
    assert pc.api_base_url({}) == "https://api.agenttrustapi.com"
    assert pc.api_base_url({"TRUST_ENGINE_API_BASE_URL": "http://localhost:9/"}) == "http://localhost:9"


# --- MCP surface -----------------------------------------------------------------------------------------------------
def _tools(handler):
    c, api = checker(handler)
    settings = dm.Settings(base_url="https://site.example", offline=True)
    return TrustEngineTools(dm.DataSource(settings), payments=c), api


def test_tool_failure_for_bad_input():
    t, _ = _tools(ok(profile(True)))
    with pytest.raises(ToolFailure, match="check_payment: amount"):
        t.check_payment(**dict(ARGS, amount="ten"))


def test_check_payment_registered_and_callable_over_mcp():
    t, api = _tools(ok(profile(True)))
    server = ms.build_server(t)
    tool = next(x for x in asyncio.run(server.list_tools()) if x.name == "check_payment")
    schema = getattr(tool, "input_schema", None) or tool.inputSchema
    assert set(schema["required"]) == {"evidence", "resource", "amount", "asset", "network"}
    assert "0.01 USDC" in tool.description and "holds no key" in tool.description
    res = asyncio.run(server.call_tool("check_payment", ARGS))
    assert '"verdict": "ALLOW"' in json.dumps(res, default=str).replace('\\"', '"') and len(api.requests) == 1
