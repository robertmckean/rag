# Canonical Schema

Phase 1 writes two canonical JSONL streams:
- `conversations.jsonl`
- `messages.jsonl`

`content_blocks` is authoritative. `text` is optional derived plain text.

## Conversation Record

Required fields:
- `conversation_id`
- `provider`
- `source_conversation_id`
- `message_count`
- `source_artifact`
- `source_metadata`

Optional fields:
- `title`
- `summary`
- `created_at`
- `updated_at`
- `participant_summary`

Shape:

```json
{
  "conversation_id": "chatgpt:conversation:conv-1",
  "provider": "chatgpt|claude",
  "source_conversation_id": "conv-1",
  "title": "string | null",
  "summary": "string | null",
  "created_at": "ISO-8601 UTC string | null",
  "updated_at": "ISO-8601 UTC string | null",
  "message_count": 5,
  "participant_summary": {
    "roles": ["assistant", "user"],
    "authors": ["User"]
  },
  "source_artifact": {
    "root": "History_ChatGPT | History_Claude",
    "conversation_file": "conversations-000.json | conversations.json",
    "sidecar_files": []
  },
  "source_metadata": {
    "conversation_origin": "string | null",
    "is_archived": "boolean | null",
    "is_starred": "boolean | null",
    "is_read_only": "boolean | null",
    "is_do_not_remember": "boolean | null",
    "is_study_mode": "boolean | null",
    "default_model_slug": "string | null",
    "account_uuid": "string | null"
  }
}
```

## Message Record

Required fields:
- `message_id`
- `conversation_id`
- `provider`
- `source_message_id`
- `parent_message_id`
- `sequence_index`
- `content_blocks`
- `source_artifact`
- `source_metadata`

Optional fields:
- `author_role`
- `author_name`
- `sender`
- `created_at`
- `updated_at`
- `text`
- `attachments`

Shape:

```json
{
  "message_id": "chatgpt:message:msg-1",
  "conversation_id": "chatgpt:conversation:conv-1",
  "provider": "chatgpt|claude",
  "source_message_id": "msg-1",
  "parent_message_id": "chatgpt:message:parent-msg | null",
  "sequence_index": 0,
  "author_role": "assistant | user | system | tool | null",
  "author_name": "string | null",
  "sender": "human | assistant | null",
  "created_at": "ISO-8601 UTC string | null",
  "updated_at": "ISO-8601 UTC string | null",
  "text": "derived plain text | null",
  "content_blocks": [
    {
      "type": "text | image | image_asset_pointer | ...",
      "text": "string | null",
      "start_timestamp": "ISO-8601 UTC string | null",
      "stop_timestamp": "ISO-8601 UTC string | null",
      "citations": []
    }
  ],
  "attachments": [
    {
      "kind": "attachment | file | image_asset_pointer | ...",
      "path": "string | null",
      "source_ref": "string | null"
    }
  ],
  "source_artifact": {
    "conversation_file": "conversations-000.json | conversations.json",
    "raw_message_path": "mapping.node-id.message | chat_messages[0]"
  },
  "source_metadata": {
    "content_type": "string | null",
    "channel": "string | null",
    "status": "string | null",
    "flags": []
  }
}
```

## Authoritative vs Derived Fields

Authoritative:
- `content_blocks`
- `attachments`
- `source_artifact`
- `source_metadata`

Derived:
- `text`
  Flattened plain text convenience field only.
- `message_count`
  Derived from emitted canonical messages.
- `participant_summary`
  Lightweight summary derived from emitted canonical messages.

## Provenance Fields

Conversation provenance:
- `provider`
- `source_conversation_id`
- `source_artifact.root`
- `source_artifact.conversation_file`

Message provenance:
- `provider`
- `source_message_id`
- `source_artifact.conversation_file`
- `source_artifact.raw_message_path`

Provider-qualified IDs remain stable across deterministic reruns for the same raw inputs.
