from typing import Any

from app.models.collections import USERS
from app.repositories.base import MongoRepository


class UserRepository(MongoRepository):
    collection_name = USERS

    async def find_by_email(self, email: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"email": email.lower()})
