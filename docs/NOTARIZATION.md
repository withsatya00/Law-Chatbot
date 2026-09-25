# E-Notarization: Preparation and Verification

## What this module does, and what it deliberately does not

This module **prepares documents for notarization** and **records the outcome
of a notarization performed by a licensed human notary**. It does not notarize
anything itself, and it contains no mechanism that could.

There is no code path anywhere in `app/notarization/` that fabricates a notary
stamp, a notary signature, a registration number, a certificate, or a
notarization record. Requests to add one should be refused. The only way a
document reaches the `notarized` state is
`NotarizationService.approve_request()`, which requires a **verified, active
notary account** to explicitly approve a specific document hash after a fresh
re-authentication.

## The six statuses that must never be conflated

| # | Thing | Status | Label shown to users |
|---|---|---|---|
| 1 | AI-generated draft | `draft` | "AI-generated draft" |
| 2 | User-signed document | (drafting module) | — |
| 3 | E-signed document | `signed` | **"Signed — not notarized"** |
| 4 | Notary-ready document | `ready_for_signature` | "Notary-ready draft" |
| 5 | Notarization requested | `notary_review_requested` | "Notarization requested" |
| 6 | Notarized document | `notarized` | **"Notarized"** |

Only `notarized` may ever be labelled "Notarized". This is enforced in one
place — `STATUS_LABELS` in `app/notarization/service.py` — and asserted by
`test_only_the_notarized_status_is_labelled_notarized`.

A PDF or DOCX is **never** labelled notarized merely because it was generated,
digitally signed, or uploaded.

## State machine

```
draft
  → ready_for_signature
      → signing_in_progress
          → signed                     (e-sign completed)
          → ready_for_signature        (failed / expired / cancelled)
      → draft                          (edited)
  → signed
      → notary_review_requested
          → notary_review_in_progress
              → notarized              (verified notary approved)
              → signed                 (REJECTED — stays non-notarized)
          → signed
  → notarized
      → revoked                        (terminal)
```

Rules the machine enforces (`app/notarization/states.py`):

- There is **no edge** from any user-reachable state directly to `notarized`.
  Only `notary_review_in_progress → notarized` exists.
- A failed, expired, or cancelled signing attempt falls **back** to
  `ready_for_signature`. It never leaves a document parked in a state that
  reads as further along than it is.
- A rejected notarization returns the document to `signed`. It never touched
  `notarized`.
- `revoked` is terminal. A replacement is a new version, not a resurrection.
- Editing a `signed` or `notarized` document is impossible in place;
  `create_version()` creates a **new document row at `draft`** and marks the
  prior version `notarization_valid: false`.

## Document integrity

Every document version carries a **SHA-256** over its rendered sections
(`integrity.hash_sections`), serialized deterministically so reordering or a
heading/body boundary shift cannot collide.

Three places compare hashes, all constant-time (`integrity.hashes_match`):

1. **E-sign callback** — if the document changed between initiation and
   callback, the signature is refused and the session is marked failed.
2. **Notary approval** — the notary re-enters the hash they reviewed. A
   mismatch refuses approval, so a notary can never attest bytes they did not
   see.
3. **Public verification** — a verifier may optionally supply the hash of the
   file they hold, and is told whether it matches.

## Audit trail

`notarization_audit_events` is **append-only at the application level**:
`NotarizationAuditRepository` deliberately does not inherit `MongoRepository`,
so it exposes `append`, `list_for_document` and `list_recent` and *no*
`update_by_id` or `delete_by_id`. There is no code path in this application
that can alter or remove a recorded event.

> **Operator action required for full immutability.** Application-level
> protection is one half. For a tamper-evident trail, grant the application's
> database user `insert` and `find` but **not** `update` or `remove` on
> `notarization_audit_events`, or place the collection on WORM storage. This
> is not something the application can do for itself.

Every event carries `occurred_at`, `recorded_at`, `actor_id`, `actor_role`,
`document_id`, `document_version`, `document_hash`, and `action`.

Recorded actions: `draft_generated`, `user_edited`, `document_exported`,
`notary_ready_prepared`, `esign_initiated`, `esign_completed`, `esign_failed`,
`notarization_requested`, `notary_review_started`, `notary_approved`,
`notary_rejected`, `notarized_version_issued`, `notarization_revoked`,
`notary_account_verified`, `notary_account_revoked`, `verification_viewed`.

`audit.redact_detail()` strips any key containing `aadhaar`, `otp`,
`biometric`, `password`, `secret`, `api_key`, `token`, `credential`,
`identity_number`, or `content` — recursively — before anything is written.

## Privacy

**Never stored, anywhere in this module:**

- Aadhaar or any other identity document **number** (only the *type* is
  collected; the notary inspects the original in person)
- OTPs
- Biometric data
- E-sign provider credentials

The `SigningSession` dataclass has nowhere to put them, so an integration
cannot casually persist them through this path.

**The public verification endpoint** (`GET /verify/{token}`) returns only:
document status, document type/title, hash-match result, notarization date,
notary name and registration number, and whether it was revoked. It returns
**no** document content, signer address or contact details, identity document
information, owning-user identity, or internal ids. Asserted by
`test_verification_returns_no_personal_data`.

The verification token itself is 32 bytes of `secrets` entropy and is **not
derived** from the document, hash, user, or notary — it carries no information
and cannot be reversed.

## Security

| Control | Where |
|---|---|
| Role-based permissions (user / verified notary / admin) | `api/deps.require_roles`, `service._require_verified_notary` |
| Ownership check on every document endpoint | `service._require_ownership` (anonymous is refused outright) |
| Re-authentication before approve/reject/revoke | `api/notarization._require_reauthentication` |
| Signed, expiring action tokens scoped to purpose **and** subject | `notarization/tokens.py` |
| Rate limits on verification and signing | `api/notarization._enforce_rate_limit` |
| Identity document numbers masked | Not collected at all — only the type |
| Constant-time hash comparison | `integrity.hashes_match` |

Action tokens are HMAC-SHA256 over a compact payload with explicit expiry, and
are validated for signature, expiry, purpose, **and** subject — so a token
issued to approve request A cannot be replayed to approve request B, nor an
approve token reused to revoke.

## E-sign providers

No provider is hardcoded. `ESIGN_PROVIDER` names a class registered in
`app/notarization/esign/registry.py`.

The default, `manual`, applies **no** electronic signature and says so
explicitly. It never reports `signed` and has no code path that can. This is
deliberate: a deployment with no vendor integration gets honest behaviour
rather than a fabricated signature.

To integrate a vendor:

```python
# app/notarization/esign/acme.py
from app.notarization.esign.base import ESignProvider, SigningRequest, SigningSession
from app.notarization.esign.registry import register_provider

@register_provider
class AcmeESignProvider(ESignProvider):
    name = "acme"

    async def initiate(self, request: SigningRequest) -> SigningSession: ...
    async def fetch_status(self, provider_reference: str) -> SigningSession: ...
    async def cancel(self, provider_reference: str) -> SigningSession: ...
```

Then set `ESIGN_PROVIDER=acme`. Nothing in the service, API, or state machine
changes.

Providers must **return** a `SigningSession` with status `failed` for ordinary
vendor errors rather than raising, and must not place identity numbers, OTPs,
biometric payloads, or credentials in `SigningSession.metadata`.

## QR codes

A verification QR is generated **only** for a `notarized` document —
`build_verification_qr_svg` raises `QRUnavailableError` for every other
status, including `revoked`. It encodes the public verification URL and
nothing else.

Encoding uses `segno` (pure Python, no native dependencies). An earlier
revision hand-rolled the encoder; it produced structurally plausible output
that no decoder could read, which is worse than no QR because it looks like a
working one. `test_qr_code_actually_decodes_to_the_verification_url` decodes
the generated code with OpenCV to prove it scans.

## API

| Method | Path | Who |
|---|---|---|
| POST | `/notarization/prepare` | owner |
| GET | `/notarization/eligibility` | any |
| GET | `/notarization/signing/providers` | authenticated |
| POST | `/notarization/signing/initiate` | owner (rate-limited) |
| POST | `/notarization/signing/callback` | provider (action token required) |
| POST | `/notarization/requests` | owner |
| GET | `/notarization/requests` | owner / verified notary |
| POST | `/notarization/requests/{id}/start-review` | verified notary |
| POST | `/notarization/requests/{id}/decision-token` | verified notary (password) |
| POST | `/notarization/requests/{id}/approve` | verified notary (action token) |
| POST | `/notarization/requests/{id}/reject` | verified notary (action token) |
| POST | `/notarization/requests/{id}/revoke` | verified notary (action token) |
| GET | `/notarization/documents/{id}/status` | owner |
| GET | `/notarization/documents/{id}/download` | owner, **notarized only** |
| GET | `/verify/{verification_token}` | **public**, rate-limited |
| POST | `/notarization/admin/notaries` | admin |
| POST | `/notarization/admin/notaries/{id}/verify` | admin |
| POST | `/notarization/admin/notaries/{id}/revoke` | admin |
| GET | `/notarization/admin/notaries` | admin |
| GET | `/notarization/admin/audit` | admin |
| GET | `/notarization/admin/failed-attempts` | admin |
| GET | `/notarization/admin/esign-config` | admin |

## Setup

### 1. Dependencies

```bash
pip install segno            # verification QR codes (pure Python)
```

`segno` is bound at call time through `optional_deps`, so a deployment without
it loses the QR on the notarized PDF and is told what to install, rather than
failing to import.

### 2. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ESIGN_PROVIDER` | `manual` | Registered provider name. `manual` applies no signature. |
| `ESIGN_SESSION_TTL_MINUTES` | `60` | How long a signing session stays valid. |
| `NOTARY_ACTION_TOKEN_TTL_SECONDS` | `300` | Lifetime of an approve/reject/revoke token. |
| `VERIFICATION_RATE_LIMIT_PER_MINUTE` | `30` | Per-IP limit on `/verify/{token}`. |
| `SIGNING_RATE_LIMIT_PER_MINUTE` | `10` | Per-IP limit on signing endpoints. |
| `VERIFICATION_BASE_URL` | `http://localhost:8000` | Public base URL the QR points at. **Must be set in production.** |

`JWT_SECRET_KEY` is reused to sign action tokens; its production strength is
already enforced by `Settings._validate_security`.

### 3. Indexes

Applied automatically at startup by `ensure_notarization_indexes()`. Idempotent.

Two are **correctness constraints**, not performance tuning:

- `notarization_documents.verification_token` — unique, sparse
- `notary_accounts.registration_number` — unique

### 4. Creating the first notary

1. The notary registers an ordinary user account.
2. An admin `POST`s to `/notarization/admin/notaries` with their name,
   registration number and jurisdiction. **The account is created
   `unverified` and cannot notarize anything.**
3. The admin verifies the notary's credentials **out of band** against the
   relevant Bar Council / State Government notary register.
4. Only then does the admin `POST` to
   `/notarization/admin/notaries/{id}/verify`.

Step 3 is a human responsibility this software cannot perform. Verifying an
account in the UI is an assertion by the administrator that they have checked
the notary's licence.

## Migration and rollback

**Migration is additive.** Every collection is new; no existing collection or
document is modified. There is no backfill.

Installing: deploy, and the startup hook creates the indexes.

Rolling back: drop the five collections listed under "E-Notarization" in
`app/models/collections.py`. Nothing outside `app/notarization/` reads them.
An older build ignores them entirely.

## Legal limitations — read before deploying

1. **This software does not notarize documents.** It records that a licensed
   notary said they did. The legal effect of that notarization depends
   entirely on the notary's licence, the applicable Notaries Act and rules,
   and the law of the relevant jurisdiction.

2. **Electronic notarization is not uniformly recognised in India.** Whether a
   remote or electronic notarial act is valid depends on the state, the
   document type, and the purpose. Several document classes (certain property
   instruments, some powers of attorney, documents for use abroad) require
   physical presence, a physical seal, stamp duty, or consular/apostille
   attestation that this platform neither performs nor replaces.

3. **A verification QR proves only what this platform recorded.** It confirms
   that a notary account marked as verified on this platform approved a
   document with a matching hash on a given date. It is not itself a notarial
   act and does not substitute for the notary's own seal and signature on the
   physical document. The printed block says exactly this.

4. **Digital signing is not notarization.** An e-signature under the
   Information Technology Act, 2000 and a notarial attestation under the
   Notaries Act, 1952 are different things with different legal effects. The
   platform states this at every signing step.

5. **Stamp duty is out of scope.** Nothing in this module assesses, collects,
   or evidences stamp duty, which is a separate statutory requirement for many
   notarizable instruments.

6. **Administrator verification is a human act.** Marking a notary "verified"
   records that an administrator checked their licence. The software performs
   no independent check against any government register.

## Remaining external integrations

- **E-sign vendor.** The `manual` provider applies no signature. A real
  integration (Digio, Leegality, eMudhra, DocuSign, NSDL, or an internal
  eSign/ASP service) must be written against `ESignProvider` and registered.
- **Notary register verification.** Administrator verification is manual.
  Automated checking against a Bar Council or State Government notary register
  would require an integration that does not exist here.
- **Aadhaar eSign / eKYC.** Not integrated, and integrating it would require
  careful handling of data this module currently refuses to store at all.
- **Payment / stamp duty.** Not integrated.
