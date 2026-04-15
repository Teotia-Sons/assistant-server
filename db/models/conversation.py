from datetime import datetime, timezone

from mongoengine import DateTimeField, Document, ListField, StringField


class Conversation(Document):
    username = StringField(required=True)
    title = StringField()
    messages = ListField()
    created_at = DateTimeField(
        default=lambda: datetime.now(timezone.utc), required=True
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "title": self.title,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
        }
