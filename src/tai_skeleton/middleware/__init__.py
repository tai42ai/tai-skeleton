"""App-level ASGI middleware that ships with the skeleton.

``RateLimitMiddleware`` (the public-door flood limiter) is registered at app
construction and is always on — tune or disable each family via
``TAI_RATE_LIMIT_*``. A manifest ``middlewares_modules`` entry is the opt-in path
for OTHER middleware you add; importing such a module fires its
``@tai_app.http.middleware`` registration.
"""
