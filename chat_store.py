from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).parent / "data" / "chat_history.db"


@dataclass
class Message:
    role: str
    content: str
    sources: list[dict] | None = None
    attachments: list[str] | None = None   # filenames attached with this message


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    doc_ids: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)


class ChatStore:
    # Manages persistent chat history with SQLite.
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New chat',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                doc_ids     TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources         TEXT,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id, id);
        """)
        # Older databases don't have the attachments column yet.
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(messages)")}
        if "attachments" not in cols:
            self.conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.commit()

    # Conversation CRUD
    def create_conversation(self, title: str = "New chat",
                            doc_ids: list[str] | None = None) -> str:
        conv_id = uuid.uuid4().hex[:12]
        now = _now()
        self.conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, doc_ids) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, title, now, now, json.dumps(doc_ids or [])),
        )
        self.conn.commit()
        return conv_id

    def list_conversations(self, search: str | None = None) -> list[Conversation]:
        # Return conversations newest-first.  If *search* is given, filter by title or message content (case-insensitive substring match).
        if search:
            rows = self.conn.execute(
                """
                SELECT DISTINCT c.*
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.title LIKE ? OR m.content LIKE ?
                ORDER BY c.updated_at DESC
                """,
                (f"%{search}%", f"%{search}%"),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_conv(r) for r in rows]

    def get_conversation(self, conv_id: str) -> Conversation | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not row:
            return None
        conv = self._row_to_conv(row)
        msg_rows = self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
        conv.messages = [
            Message(
                role=m["role"],
                content=m["content"],
                sources=json.loads(m["sources"]) if m["sources"] else None,
                attachments=json.loads(m["attachments"]) if m["attachments"] else None,
            )
            for m in msg_rows
        ]
        return conv

    def update_title(self, conv_id: str, title: str):
        self.conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conv_id),
        )
        self.conn.commit()

    def update_doc_ids(self, conv_id: str, doc_ids: list[str]):
        self.conn.execute(
            "UPDATE conversations SET doc_ids = ?, updated_at = ? WHERE id = ?",
            (json.dumps(doc_ids), _now(), conv_id),
        )
        self.conn.commit()

    def get_doc_ids(self, conv_id: str) -> list[str]:
        row = self.conn.execute(
            "SELECT doc_ids FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        return json.loads(row["doc_ids"]) if row else []

    def doc_in_use(self, doc_id: str) -> bool:
        # True if any conversation still has this PDF attached.
        return any(doc_id in json.loads(r["doc_ids"])
                   for r in self.conn.execute("SELECT doc_ids FROM conversations"))

    def delete_conversation(self, conv_id: str):
        self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        self.conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        self.conn.commit()

    # Messages
    def add_message(self, conv_id: str, role: str, content: str,
                    sources: list[dict] | None = None,
                    attachments: list[str] | None = None):
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, attachments, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, role, content,
             json.dumps(sources) if sources else None,
             json.dumps(attachments) if attachments else None, _now()),
        )
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conv_id),
        )
        self.conn.commit()

    def clear_messages(self, conv_id: str):
        # Delete all messages in a conversation but keep the conversation.
        self.conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conv_id,)
        )
        self.conn.commit()

    # Helpers

    @staticmethod
    def _row_to_conv(row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            doc_ids=json.loads(row["doc_ids"]),
        )

    def close(self):
        self.conn.close()


def auto_title(content: str, max_len: int = 50) -> str:
    # Derive a short title from the first user message.
    title = content.strip().split("\n")[0]
    if len(title) > max_len:
        title = title[: max_len - 1] + "…"
    return title or "New chat"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
