# canvas/data.json

This file feeds a live status badge on [evintkoo.github.io](https://evintkoo.github.io)'s mindmap ("canvas") card for this project. Edit it and push to `main` — the site picks up the change on next page load, no rebuild needed on the portfolio side.

## Schema

```json
{
  "status": "planned" | "in-progress" | "testing" | "done",
  "title": "optional override of the card's title",
  "description": "optional override of the card's description"
}
```

- `status` — required. One of `planned`, `in-progress`, `testing`, `done`.
- `title` — optional. Overrides the title shown on the portfolio card.
- `description` — optional. Overrides the description shown on the portfolio card.

## Notes

- Must live at `canvas/data.json` on this repo's `main` branch — the site fetches from `main` only, no fallback to other branches.
- The site fetches this file directly and anonymously from `raw.githubusercontent.com`, so it only works for **public** repos.
