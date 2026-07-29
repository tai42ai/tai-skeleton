"""App-level ASGI middleware that ships with the skeleton.

``RateLimitMiddleware`` (the public-door flood limiter) is registered at app
construction and is always on — tune or disable each family via
``TAI_RATE_LIMIT_*``. ``AuditLogMiddleware`` (the one-line-per-request audit
trail) and ``BodyLimitMiddleware`` (the body-size cap) are registered on the base
app, INSIDE the access-control gate, so the audit line carries the identity the
gate resolved; the audit trail is on by default and ``TAI_AUDIT_LOG_ENABLE=false``
leaves it unregistered. A manifest ``middlewares_modules`` entry is the opt-in
path for OTHER middleware you add; importing such a module fires its
``@tai42_app.http.middleware`` registration.
"""
