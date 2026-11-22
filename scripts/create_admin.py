"""Create or promote an admin account: the only way to grant the admin role (blueprint §5.8.1).

Usage:
    .venv\\Scripts\\python scripts\\create_admin.py admin@example.org [--display-name "Ops"]
    .venv\\Scripts\\python scripts\\create_admin.py admin@example.org --demote

- An existing account is promoted to admin (its password is unchanged and not asked for).
- A new account is created as admin; the password is read with `getpass` (twice), never echoed
  and never accepted as a command-line argument, so it stays out of shell history.
- `--demote` turns an admin back into a user and bumps `token_version`, so the old tokens die (E42).
  The last admin cannot be demoted.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.core.auth_service import AuthService
from app.core.errors import AppError
from app.db.session import Database

MIN_PASSWORD_CHARS = 10


def prompt_password(read: Callable[[str], str] = getpass.getpass) -> str:
    first = read("New admin password: ")
    if len(first) < MIN_PASSWORD_CHARS:
        raise SystemExit(f"The password must be at least {MIN_PASSWORD_CHARS} characters.")
    if read("Repeat the password: ") != first:
        raise SystemExit("The passwords do not match.")
    return first


async def run(
    email: str,
    *,
    display_name: str | None,
    demote: bool,
    settings: Settings,
    read_password: Callable[[str], str] = getpass.getpass,
) -> int:
    db = Database(settings.database_url)
    await db.create_all()
    try:
        auth = AuthService(settings, db)
        if demote:
            user = await auth.demote_admin(email)
            print(f"Demoted {user.email} to user; their existing tokens are revoked.")
            return 0
        existing = await auth.find_user(email)
        password = None if existing else prompt_password(read_password)
        user, created = await auth.create_or_promote_admin(email, password, display_name=display_name)
        print(f"{'Created' if created else 'Promoted'} admin {user.email} ({user.id}).")
        return 0
    except AppError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        return 1
    finally:
        await db.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or promote an admin account.")
    parser.add_argument("email")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--demote", action="store_true", help="turn an admin back into a user")
    args = parser.parse_args(argv)
    return asyncio.run(
        run(args.email, display_name=args.display_name, demote=args.demote, settings=Settings())
    )


if __name__ == "__main__":
    raise SystemExit(main())
