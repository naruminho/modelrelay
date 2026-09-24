from __future__ import annotations

import os
import threading
import time
from typing import Callable

import httpx

from ._util import dig, log
from .errors import AuthError, ConfigError


class TokenProvider:
    """Gives out a valid bearer token. `refreshable` providers get a second chance on 401/403."""

    refreshable = False

    def get_token(self) -> str:
        raise NotImplementedError

    def invalidate(self) -> None:
        pass


class StaticToken(TokenProvider):
    """A fixed API key. It never expires."""

    def __init__(self, token: str | None, missing_hint: str = ""):
        self._token = token
        self._missing_hint = missing_hint

    def get_token(self) -> str:
        if not self._token:
            raise ConfigError(f"No API key configured. {self._missing_hint}".strip())
        return self._token


class ClientCredentials(TokenProvider):
    """Exchanges a client id + secret for a token that lasts `ttl_minutes`.

    The token is renewed `refresh_margin_seconds` before it expires. The endpoint's
    field names are configurable; subclass and override `fetch_token` when the
    exchange is more unusual than that.
    """

    refreshable = True

    def __init__(
        self,
        http: httpx.Client,
        *,
        token_url: str,
        client_id: str | None,
        client_secret: str | None,
        ttl_minutes: float = 30,
        refresh_margin_seconds: float = 120,
        request_format: str = "json",
        id_field: str = "client_id",
        secret_field: str = "client_secret",
        token_field: str = "access_token",
        extra_fields: dict | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if request_format not in ("json", "form"):
            raise ConfigError("auth_options.request_format must be 'json' or 'form'")
        self.http = http
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.ttl = ttl_minutes * 60
        self.margin = min(refresh_margin_seconds, self.ttl / 2)
        self.request_format = request_format
        self.id_field = id_field
        self.secret_field = secret_field
        self.token_field = token_field
        self.extra_fields = extra_fields or {}
        self._clock = clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

    def get_token(self) -> str:
        with self._lock:
            if self._token is None or self._clock() >= self._expires_at - self.margin:
                log.info("requesting a new token from %s", self.token_url)
                self._token = self.fetch_token()
                self._expires_at = self._clock() + self.ttl
            return self._token

    def invalidate(self) -> None:
        with self._lock:
            self._token = None

    def fetch_token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise ConfigError(
                "client_credentials needs a client id and secret: set auth_options.client_id and "
                "auth_options.client_secret in the config file (or $MODELRELAY_CLIENT_ID / $MODELRELAY_CLIENT_SECRET)."
            )
        body = {self.id_field: self.client_id, self.secret_field: self.client_secret, **self.extra_fields}
        kwargs = {"json": body} if self.request_format == "json" else {"data": body}
        try:
            r = self.http.post(self.token_url, **kwargs)
        except httpx.HTTPError as e:
            raise AuthError(f"Could not reach the token endpoint: {type(e).__name__}: {e}", url=self.token_url) from e
        if r.status_code >= 400:
            raise AuthError(f"Token endpoint answered HTTP {r.status_code}: {r.text[:500]}",
                            url=self.token_url, status=r.status_code)
        try:
            data = r.json()
        except ValueError as e:
            raise AuthError(f"Token endpoint did not return JSON: {r.text[:500]!r}", url=self.token_url) from e
        token = dig(data, self.token_field)
        if not token:
            fields = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise AuthError(f"Token endpoint response has no '{self.token_field}' field (top-level fields: {fields})",
                            url=self.token_url)
        return token


class BearerAuth(httpx.Auth):
    """Adds the token to every request; on 401/403 renews it once and retries."""

    def __init__(self, provider: TokenProvider):
        self.provider = provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.provider.get_token()}"
        response = yield request
        if response.status_code in (401, 403) and self.provider.refreshable:
            log.info("HTTP %s on %s %s: renewing the token and retrying once",
                     response.status_code, request.method, request.url)
            self.provider.invalidate()
            request.headers["Authorization"] = f"Bearer {self.provider.get_token()}"
            yield request


BUILTIN_AUTH = {"static": StaticToken, "client_credentials": ClientCredentials}


def build_token_provider(config, http: httpx.Client) -> TokenProvider:
    from ._util import load_object

    options = dict(config.auth_options)
    if config.auth == "static":
        key = config.api_key or os.environ.get(config.api_key_env)
        where = config.source or "built in code"
        return StaticToken(key, missing_hint=f"Set `api_key`, or the {config.api_key_env} environment variable. Config: {where}")
    if config.auth == "client_credentials":
        if "token_url" not in options:
            raise ConfigError("auth_options.token_url is required for client_credentials.")
        # Values in the file win; the environment variables are only a fallback.
        id_env = options.pop("client_id_env", "MODELRELAY_CLIENT_ID")
        secret_env = options.pop("client_secret_env", "MODELRELAY_CLIENT_SECRET")
        return ClientCredentials(
            http,
            client_id=options.pop("client_id", None) or os.environ.get(id_env),
            client_secret=options.pop("client_secret", None) or os.environ.get(secret_env),
            **options,
        )
    cls = load_object(config.auth, "modelrelay.auth", BUILTIN_AUTH)
    return cls(http, **options)
