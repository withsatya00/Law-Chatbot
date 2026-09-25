"""Shared client-IP resolution for every per-request rate limiter.

Extracted out of `app.core.middleware.RateLimitMiddleware` so the
notarization flows' own tighter rate limits (`app/api/notarization.py`)
apply the same trusted-proxy rule instead of trusting `request.client.host`
unconditionally -- before this, only the global middleware had the check.
"""
from fastapi import Request

from app.core.config import settings


def client_ip(request: Request) -> str:
    """The address this request should be rate-limited on.

    The TCP peer (`request.client.host`) is always correct in a direct
    deployment. Behind a reverse proxy/load balancer, that peer is the proxy
    itself, so every real client would collapse onto one bucket -- but
    `X-Forwarded-For` is only trusted when the peer is a configured proxy
    (`settings.trusted_proxy_ips`); from any other peer it's attacker-supplied
    and would let one client spoof another's IP to dodge or frame their limit.
    """
    peer = request.client.host if request.client else "unknown"
    if peer in settings.trusted_proxy_ips:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Left-most entry is the original client per the de-facto
            # `X-Forwarded-For` convention (each hop appends its own peer).
            candidate = forwarded.split(",")[0].strip()
            if candidate:
                return candidate
    return peer
