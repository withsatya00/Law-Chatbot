# Deployment Guide

## Targets

The service can run on AWS, Azure, Google Cloud, Render, Railway, DigitalOcean, or a Linux VPS as long as the deployment provides:

- Python 3.12 runtime or Docker
- MongoDB Atlas or MongoDB service
- Redis
- HTTPS ingress
- Secret management for `.env`

## Required Steps

1. Provision MongoDB and Redis.
2. Configure `.env` from `.env.example`.
3. Build and deploy the API container.
4. Run `python scripts/create_indexes.py`.
5. Deploy Streamlit only for internal testing.
6. Configure health checks against `/health`.
7. Enable log forwarding and alerting.

## Security Checklist

- Rotate `JWT_SECRET_KEY`.
- Restrict `API_CORS_ORIGINS`.
- Set upload limits.
- Scan uploaded files before permanent storage.
- Keep prompt logs access-controlled because they may contain user-provided facts.
- Disable Streamlit in public production environments.
