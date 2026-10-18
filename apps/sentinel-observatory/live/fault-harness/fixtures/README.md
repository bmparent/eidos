# Test-only TLS identity

These public fixture credentials authenticate only the loopback fault harness.
The certificate covers example.com/example.org and is explicitly trusted by
the test adapter. Never use this key or CA for a hosted service. Production
does not import this directory and always validates the original target identity.

Generated with OpenSSL on 2026-10-18, valid for ten years. Certificate lifetime
is recorded in its PEM; these files are test inputs, not operating credentials.
