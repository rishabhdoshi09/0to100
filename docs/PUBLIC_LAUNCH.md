# QuantTerm public internet launch

QuantTerm's canonical runtime stays bound to loopback. Do **not** expose ports
`5173`, `8765`, or `8766` directly to the internet. Put a TLS reverse proxy
or another deployment edge in front of the public read-only experience.

## Required public mode

Set this in the environment inherited by the QuantTerm API process:

```bash
QT_PUBLIC_READ_ONLY=1
```

With public mode enabled:

- GET/HEAD/OPTIONS reads continue to work.
- POST/PUT/PATCH/DELETE requests are rejected with HTTP 403 by default.
- The guard applies to the base terminal routes and product-extension routes on
  the same FastAPI app, including scan/data/F&O refreshes, paper/autonomy
  controls, watchlist mutations, bootstrap, and due-diligence acquisition.
- The guard does not trust client IPs or `X-Forwarded-For`. A local reverse
  proxy therefore cannot accidentally make an internet visitor an operator.
- This boundary does not enable live money. QuantTerm's existing live-money
  interlock remains separate and locked.

Verify the effective posture with:

```text
GET /api/access
GET /api/health
```

A public visitor deployment should report `public_read_only: true`.

## Optional operator mutations

If the same API instance must accept authenticated operator mutations, provide a
high-entropy secret only to the API process:

```bash
QT_PUBLIC_READ_ONLY=1
QT_OPERATOR_TOKEN=<high-entropy-secret>
```

Unsafe requests then require:

```http
Authorization: Bearer <high-entropy-secret>
```

Never place `QT_OPERATOR_TOKEN` in frontend source, a `VITE_*` variable,
browser localStorage, query parameters, URLs, logs, screenshots, or public
deployment configuration. Prefer a private operator path at the reverse proxy
or a separate private admin client that injects the header server-side.

If `QT_OPERATOR_TOKEN` is absent while public mode is enabled, QuantTerm is
strictly read-only. That is the recommended public launch posture.

## Reverse-proxy rules

The deployment edge should:

1. terminate HTTPS;
2. proxy the public frontend and read-only API routes to the loopback services;
3. preserve the API's 403 responses;
4. rate-limit expensive public reads where appropriate;
5. never expose filesystem paths, runtime databases, environment files, logs,
   or the operator bearer token;
6. never rewrite blocked POST/PUT/PATCH/DELETE requests into GET requests.

For the safest first launch, expose only the public read experience and keep all
operator mutations on the private/local QuantTerm desk.
