from app.core.exceptions import BadRequestError, UnauthorizedError
from app.core.security import (
    Role,
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.repositories.users import UserRepository
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse


class AuthService:
    def __init__(self, users: UserRepository | None = None) -> None:
        self.users = users or UserRepository()

    async def register(self, request: RegisterRequest) -> TokenResponse:
        """Public self-registration always creates a `Role.user` account --
        role is never client-controlled here. Privileged roles are granted
        out-of-band; see `scripts/create_privileged_user.py`.
        """
        existing = await self.users.find_by_email(request.email)
        if existing:
            raise BadRequestError("An account with this email already exists.")
        user_id = await self.users.insert(
            {
                "email": request.email.lower(),
                "full_name": request.full_name,
                "password_hash": hash_password(request.password),
                "role": Role.user.value,
                "is_active": True,
            }
        )
        return self._tokens(user_id, Role.user)

    async def login(self, request: LoginRequest) -> TokenResponse:
        user = await self.users.find_by_email(request.email)
        if not user or not verify_password(request.password, user["password_hash"]):
            raise UnauthorizedError("Invalid email or password.")
        if not user.get("is_active", True):
            raise UnauthorizedError("Account is inactive.")
        return self._tokens(str(user["_id"]), Role(user.get("role", Role.user.value)))

    def _tokens(self, user_id: str, role: Role) -> TokenResponse:
        return TokenResponse(
            access_token=create_access_token(user_id, role),
            refresh_token=create_refresh_token(user_id, role),
            role=role,
            user_id=user_id,
        )
