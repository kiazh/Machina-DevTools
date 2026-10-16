# Deterministic skills request corpus (fixture, A0)

Inputs and required decisions for the deterministic selector/search work
(Track A). Derived from spec §8.2 ("Broad task coverage without blanket keyword
routing") and §18 S02–S11. This is a fixture corpus for A2/A6 gates, not prose.

Provenance: baseline commit `b7bd6b611e5323cf0d1db0ea3e1a9f6ae1acbb2a`,
working-tree `git diff` fingerprint
`2bf3a0bf99c9545c98a7b1b4d6e99e4bd6acda8ab2e9f561985bf31ab813c790`
(2026-10-16). See `docs/verification/DETERMINISTIC_SKILLS_A0_BASELINE.md` in the
machina checkout.

## Case schema

```json
{
  "id": "S03-make-me-a-website",
  "kind": "routing | explain | ambiguous | review | continuation | stack-override",
  "input": "<literal user message>",
  "context": { "...": "optional explicit bounded task state (only S08, S09)" },
  "required": {
    "task_family": "<family slug> | null",
    "stack_resolution": "unresolved | explicit | inherited | not_applicable",
    "constraints_preserved": ["<constraint tokens that must survive>"],
    "must_not": ["<forbidden observable decisions>"]
  }
}
```

Field semantics (consumed by A2; freeze vocabulary there):

- `task_family`: slug that the decision must assert. `null` means no positive
  family assertion is required — negatives still apply via `must_not`.
  `web-build` and `frontend-review` are the two families named in spec §7.2/§8.2;
  `python-testing` is a provisional fixture slug used only to check that the
  continuation path **inherits** the active task's family, not the specific
  spelling.
- `stack_resolution`: `unresolved` — resolution must stay unknown (no
  technology may be inferred from the words in `input`); `explicit` — the
  user-named stack (see `constraints_preserved`); `inherited` — taken from
  `context.active_task`; `not_applicable` — no implementation stack is involved.
- `constraints_preserved`: user constraints (read-only scope, explicit stack
  choice) that must remain present in the descriptor/selection/disclosure chain,
  not be erased by later build/fix matches (spec §8.3).
- `must_not`: observable forbidden decisions (lowercase slugs). Examples:
  `infer_react` (mapping "website" to React), `start_implementation`
  (launching work for an explanation), `route_to_web_build` (stock portfolio),
  `resolve_language_from_bare_token` ("go"/"rust"/"swift" alone).
- `context`: supplied turn-scoped state. Exists only so "with inherited/active
  state" cases are executable; absent means no task state is available.

## Cases (15)

Quoted-website (S06) and bare `go`/`rust`/`swift` (S11) expect an **ambiguous**
result: spec §8.3 prefers an honest ambiguous outcome over pretending to parse
quotes, and `intent.py` already never guesses bare follow-ons.

## Validation

```bash
python3 -c "import json; d=json.load(open('tests/data/deterministic_skills/requests.json')); \
assert isinstance(d, list) and len(d) == 15; \
assert all(c['kind'] in {'routing','explain','ambiguous','review','continuation','stack-override'} for c in d); \
print('requests.json OK:', len(d), 'cases')"
```
