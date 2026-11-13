# Issue tracker: Local Markdown

Issues, specs and wayfinder maps for this repo live as markdown files in
`.scratch/`, versioned in git (the directory is **not** ignored: commit the
files with the work they describe). Owner decision, 2026-11-13, for the F15
planning flow; the roadmap (`docs/ROADMAP.md`) stays the authority for task
status and ticks, and the phase plans in `docs/plans/` stay the authority for
the approved plan. Tickets are working notes that point at those documents.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/` (one per roadmap phase,
  e.g. `.scratch/f15/`)
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a
  single combined tickets file
- Triage state is recorded as a `Status:` line near the top of each issue file
  (`ready-for-agent`, `ready-for-human`, `needs-info`, `claimed`, `resolved`,
  `wontfix`)
- Comments and conversation history append to the bottom of the file under a
  `## Comments` heading
- Ticket titles cite the roadmap task IDs they serve (e.g. `F15.1`)

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if
needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or
the issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.
Decision tickets and implementation tickets of the same phase live in separate
subdirectories so their numbering never collides.

- **Map**: `.scratch/<effort>/map.md` (the Destination / Notes /
  Decisions-so-far / Not-yet-specified / Out-of-scope body).
- **Child ticket**: `.scratch/<effort>/decisions/NN-<slug>.md`, numbered from
  `01`, with the question in the body. A `Type:` line records the ticket type
  (`research`/`prototype`/`grilling`/`task`); a `Status:` line records
  `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked
  when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/decisions/` for files that are open,
  unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set
  `Status: resolved`, then append a context pointer (gist + link) to the map's
  Decisions-so-far in `map.md`.
- **Research assets**: findings files go next to the ticket
  (`.scratch/<effort>/research/<slug>.md`), committed on the phase branch
  instead of a throwaway `research/<name>` branch, so the plan cites them.
