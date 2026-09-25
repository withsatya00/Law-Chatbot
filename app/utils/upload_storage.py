"""Bounded upload writes with cleanup after the file handle has closed."""
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import BadRequestError


async def write_upload(file: UploadFile, destination: Path) -> None:
    created = False
    try:
        # Exclusive creation prevents simultaneous uploads overwriting a reserved path.
        with destination.open("xb") as handle:
            created = True
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise BadRequestError(f"Upload exceeds {settings.max_upload_mb} MB limit.")
                handle.write(chunk)
    except BaseException:
        # Includes request cancellation. Windows cannot unlink an open handle.
        if created:
            destination.unlink(missing_ok=True)
        raise
