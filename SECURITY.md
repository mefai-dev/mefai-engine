# Security Policy

## Important Upgrade Notice · 2026-04-21

Pre-`158167d` versions of this repository (before 2026-03-25) shipped a
`docker-compose.yml` that contained a hard coded fallback PostgreSQL
password, left the `5432:5432` host port mapping open, and ran Redis
without authentication. That default credential is considered
**compromised** and must not be reused.

### Action required for any deployment earlier than `158167d`

1. Upgrade to commit `158167d` or newer (the `Security hardening:` commit).
2. Rotate the PostgreSQL password to a strong per-deployment secret and set
   `DB_PASSWORD` explicitly in your environment. The current compose file
   fails fast if the variable is unset.
3. Set `REDIS_PASSWORD` and ensure Redis is not reachable without it.
4. Do not expose the `5432` or `6379` ports on the public internet. Bind to
   `127.0.0.1` only unless you understand the exposure.
5. Audit your PostgreSQL logs for unknown IPs that authenticated with the
   old password, and rebuild the database from a verified backup if any are
   found.

The leak exists in the git history of earlier commits. Treat the
original shipped default as a known burned credential and ensure it is
not accepted anywhere in your infrastructure.

## Reporting a Vulnerability

If you discover a security vulnerability in MEFAI Engine please report it responsibly.
Do not open a public GitHub issue for security vulnerabilities.

**Contact**: security@mefai.io

Please include:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes if you have them

We aim to acknowledge reports within 48 hours and provide a resolution timeline
within 5 business days.

## Scope

### In Scope

- The MEFAI Engine core library (everything under `src/mefai_engine/`)
- API endpoints exposed by the FastAPI application
- Authentication and authorization logic
- Cryptographic implementations and secret handling
- Database query construction (SQL injection)
- WebSocket connection handling
- Configuration and secret management
- Exchange API credential storage and transmission
- Multi-tenant isolation boundaries

### Out of Scope

- Vulnerabilities in upstream dependencies (report those to the relevant project)
- Social engineering attacks against MEFAI team members
- Denial of service attacks against hosted infrastructure
- Issues in example scripts or documentation
- Third party exchange API vulnerabilities

## Disclosure Policy

- We will work with you to verify and fix the vulnerability before any public disclosure
- We request a 90 day disclosure window from the initial report
- We will credit reporters in our changelog unless they prefer to remain anonymous
- We do not offer monetary bounties at this time but may do so in the future

## Security Best Practices for Users

1. Never commit API keys or secrets to version control
2. Use environment variables or a secrets manager for all credentials
3. Enable testnet mode during development and testing
4. Set conservative risk limits before enabling live trading
5. Monitor the circuit breaker and audit logs regularly
6. Keep all dependencies up to date
7. Use the principle of least privilege for database and API access
