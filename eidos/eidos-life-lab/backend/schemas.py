from typing import Any, Dict, Optional

from pydantic import BaseModel


class CommandRequest(BaseModel):
    command: str
    settings: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

    def to_payload(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump(exclude_none=True)
        return self.dict(exclude_none=True)
