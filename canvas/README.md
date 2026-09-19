# canvas/data.json

This file is the source of truth for this project's "canvas" card on [evintkoo.github.io](https://evintkoo.github.io) — its live status badge, and (via `branches` below) its ENTIRE feature-breakdown mindmap. Edit it and push to `main` — the site picks the change up on next page load, no rebuild and no change ever needed on the portfolio side.

## Schema

```json
{
  "status": "planned" | "in-progress" | "testing" | "done",
  "title": "optional override of the card's title",
  "description": "optional override of the card's description",
  "path": "optional/directory/or/file",
  "branches": [
    {
      "label": "A major area of this project",
      "description": "optional",
      "path": "optional/directory/or/file",
      "status": "optional — derived from its own nodes' worst status if omitted",
      "nodes": [
        {
          "id": "optional-stable-id",
          "title": "A specific feature/component",
          "description": "optional",
          "path": "optional/directory/or/file",
          "status": "planned" | "in-progress" | "testing" | "done"
        }
      ]
    }
  ],
  "nodes": {
    "some-node-id": { "status": "planned" | "in-progress" | "testing" | "done" }
  }
}
```

- `status` — required. The project's own overall status.
- `title` / `description` — optional. Override the title/description shown on the portfolio's root card for this project.
- `path` — optional. A directory or file within THIS repo that the root card's own "Go to repo" link should open (e.g. `"src/core"`), instead of just the bare repo root.
- `branches` — optional, but this is the actual point of this file: the WHOLE feature-breakdown mindmap the portfolio renders for this project, in order. Each branch is a major area (a "hub" card); each of its `nodes` is one specific feature/component (a "leaf" card) within it. If you publish this, you never need to touch the portfolio repo's own content again — add, remove, reorder, or edit anything here and the site reflects it on next load.
- `nodes` (top-level, flat) — optional, and only relevant if you're NOT publishing `branches` yet (e.g. the breakdown is still authored in the portfolio repo's own `.mdx` file). Lets you override an individual existing node's status/description/path by id without hosting the whole tree yourself.

### Node `id`

- Optional on every leaf. If set, other nodes' descriptions can cross-reference it, and it's the stable key other tooling (or a flat `nodes` override, above) addresses it by.
- If omitted, one is derived from the node's own title: lowercase, every run of non-alphanumeric characters collapsed to a single `-`, leading/trailing `-` trimmed. E.g. "Phase A / A-alt: Max-Spread Greedy" → `phase-a-a-alt-max-spread-greedy`.
- A branch gets an id the same way, prefixed `hub-` (e.g. "SAST — Static Source Analysis" → `hub-sast-static-source-analysis`) — this is what a flat `nodes` override (above) would use to target a BRANCH rather than a leaf.

## Notes

- Must live at `canvas/data.json` on this repo's `main` branch — the site fetches from `main` only, no fallback to other branches.
- The site fetches this file directly and anonymously from `raw.githubusercontent.com` for a **public** repo. A **private** repo's data only reaches the site at its next rebuild (via an authenticated build-time fetch) — everything above still applies, just with rebuild-latency freshness instead of instant.
