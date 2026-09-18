# canvas/data.json

This file feeds a live status badge on [evintkoo.github.io](https://evintkoo.github.io)'s mindmap ("canvas") card for this project. Edit it and push to `main` — the site picks up the change on next page load, no rebuild needed on the portfolio side.

## Schema

```json
{
  "status": "planned" | "in-progress" | "testing" | "done",
  "title": "optional override of the card's title",
  "description": "optional override of the card's description",
  "nodes": {
    "some-node-id": { "status": "planned" | "in-progress" | "testing" | "done" }
  }
}
```

- `status` — required. The project/paper's own overall status. One of `planned`, `in-progress`, `testing`, `done`.
- `title` — optional. Overrides the title shown on the portfolio card.
- `description` — optional. Overrides the description shown on the portfolio card.
- `nodes` — optional. Only relevant if this project/paper has its own methodology breakdown shown on its detail page (a paper's `breakdown` field in its portfolio content). Lets you give each individual note inside that breakdown its own status, independent of the top-level `status` above. Each entry also supports its own `title`/`description` override, same shape as the top-level fields.

### Finding a node's id

- If the note is one you've explicitly cross-referenced from elsewhere in the breakdown (the portfolio's `<inode id="...">` markup), use that same id — it's visible in the portfolio repo's `.mdx` source for this entry (`src/content/research/<slug>.mdx`, the `breakdown.branches[].nodes[].id` field).
- Otherwise, the id is derived from the note's own title: lowercase, every run of non-alphanumeric characters collapsed to a single `-`, leading/trailing `-` trimmed. E.g. "Phase A / A-alt: Max-Spread Greedy" → `phase-a-a-alt-max-spread-greedy`.
- A `nodes` entry for an id that doesn't exist in the breakdown is simply ignored — nothing breaks.

## Notes

- Must live at `canvas/data.json` on this repo's `main` branch — the site fetches from `main` only, no fallback to other branches.
- The site fetches this file directly and anonymously from `raw.githubusercontent.com`, so it only works for **public** repos.
