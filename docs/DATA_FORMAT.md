# Article data format

Put user data in `data/articles/` or pass another path with `--input`. Directories are
searched recursively in filename order. JSON, JSONL, CSV, and UTF-8 TXT are supported
and can be mixed.

## Fields

| Field | Required | Meaning |
|---|---:|---|
| `text` | yes | Complete article body; `content` is accepted as an alias |
| `id` | no | Unique stable identifier; generated from title and text if omitted |
| `title` | no | Human-readable title |
| `source` | no | Publisher shown only in the report |
| `url` | no | Source link shown only in the report |
| `published_at` | no | ISO date or datetime shown only in the report |
| `label` | no | Optional `Left`, `Center`, or `Right` evaluation label |

Unknown fields are preserved under article metadata. Metadata is never included in
the model prompt.

## JSON

One object per file or an array of objects:

```json
{
  "id": "local-42",
  "title": "Example",
  "text": "Complete article text...",
  "source": "Example News",
  "url": "https://example.com/article",
  "published_at": "2026-01-15",
  "label": "Center"
}
```

## JSONL

Each non-empty line is one article object. This is the recommended format for large
collections because it is easy to stream and version.

## CSV

The first row contains field names. The `text` or `content` column is required. CSV
quoting handles line breaks and commas in article text.

## TXT

Each `.txt` file is one article. The filename becomes the ID and title. TXT inputs
cannot supply labels or metadata.

## Validation

- Text shorter than 80 characters is rejected as likely incomplete.
- IDs must be unique across the complete input collection.
- Files must be valid UTF-8.
- Articles over `method.max_article_characters` are rejected rather than silently
  truncated. Increase this setting only when every configured model has enough
  context capacity.

Run `python3 run.py --validate-only` to check data without loading a model.
