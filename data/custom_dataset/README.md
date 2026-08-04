# Custom AllSides dataset

This folder is the home of the project's custom article collection in the public
[MADS article format](../../docs/DATA_FORMAT.md). The repository already includes:

- `../article_outlet_labels.csv`: AllSides outlet matches for 473,989 collected
  article IDs;
- `../allsides/AllSides_Rating.csv`: the AllSides 2025 outlet-rating reference.

The original article bodies are not committed because the full collection is about
3.9 GB. When the bodies are available, put their JSON files in `raw_articles/` and
convert all of them with:

```bash
python3 tools/convert_custom_dataset.py
```

This creates `articles.jsonl`. Analyze it with:

```bash
python3 run.py --input data/custom_dataset/articles.jsonl
```

Use `--limit 100` for a first real-model run. The report includes article metrics,
outlet metrics, a score-distribution figure, and an outlet-comparison figure.

## Ground-truth assumption

For this custom evaluation, we assume that every article's ground-truth label is the
AllSides 2025 rating of its media outlet. This is an **outlet-derived proxy label**,
not an independent annotation of the article's content.

The converter uses the same three-class mapping as the paper:

| AllSides 2025 outlet rating | MADS article `label` |
|---|---|
| Left or Lean Left | Left |
| Center | Center |
| Lean Right or Right | Right |
| Mixed or unmatched | no `label` field |

Mixed and unmatched articles remain in `articles.jsonl` and receive model
predictions, but they are excluded from accuracy and F1 calculations. In the
committed 473,989-row lookup table, 201,056 article IDs have a compatible
Left/Center/Right proxy label, 2,499 are Mixed, and 270,434 are unmatched.

## Historical input fields

The converter accepts the historical article fields directly:

- ID: `record_id`, `ID`, or `id`;
- body: `content_original`, `content`, or `text`;
- outlet: `source_name`, `source`, or `outlet`;
- date: `date`, `published`, or `published_at`.

Its output uses only the recommended public schema. Invalid files are reported in
`conversion_errors.csv` and are never silently included.
