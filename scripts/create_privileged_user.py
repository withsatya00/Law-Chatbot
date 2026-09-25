"""Internal-only provisioning for privileged accounts (lawyer/admin/super_admin).

The public `POST /register` endpoint always creates `role=user` accounts --
it never accepts a client-supplied role. This script is the only supported
way to create an elevated account, and is never exposed over HTTP: run it
locally by an operator with direct database access.

Usage:
    python scripts/create_privileged_user.py --email a@b.com --password "..." \
        --full-name "Jane Doe" --role admin
"""

import argparse
import asyncio

import structlog

from app.core.config import settings
from app.core.logger import configure_logging
from app.core.security import Role, hash_password
from app.database.mongodb import mongodb
from app.repositories.users import UserRepository

log = structlog.get_logger(__name__)


async def create_privileged_user(email: str, password: str, full_name: str, role: Role) -> str:
    users = UserRepository()
    existing = await users.find_by_email(email)
    if existing:
        raise SystemExit(f"An account with email {email!r} already exists.")
    user_id = await users.insert(
        {
            "email": email.lower(),
            "full_name": full_name,
            "password_hash": hash_password(password),
            "role": role.value,
            "is_active": True,
        }
    )
    return user_id


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--role", required=True, choices=[Role.lawyer.value, Role.admin.value, Role.super_admin.value])
    args = parser.parse_args()

    configure_logging()
    await mongodb.connect()
    try:
        user_id = await create_privileged_user(args.email, args.password, args.full_name, Role(args.role))
        log.info("privileged_user_created", user_id=user_id, email=args.email, role=args.role, environment=settings.environment)
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
