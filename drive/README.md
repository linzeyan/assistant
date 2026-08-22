# Driving the assistant through a coding task

A way of using this backend as an unattended pair of hands on a real codebase, and
the scripts that make it repeatable. You decide every line of the change; the model
places it, runs the build and the tests, and reports what happened. One brief, one
turn, one reviewable diff.

It is not autonomous coding. Everything here exists because the local models this
project runs are *very* good at one half of the job and unreliable at the other,
and the split is sharper than it looks:

| The model is good at | The model is bad at |
|---|---|
| applying an exact edit at an exact place | deciding what the edit should be |
| running builds, tests, formatters | deciding what to do when one fails |
| reporting output verbatim | judging whether that output means success |

So the method is: **write the finished code into the brief, and leave only
placement and gate-running to the model.** Every rule below is one measured
failure, with the observation that produced it.

```
    you                          drive.py                     the model
 ┌────────────┐   brief.md    ┌────────────┐   POST /chat   ┌────────────┐
 │ decide the │──────────────▶│  dispatch  │───────────────▶│  edit_file │
 │  change    │               │  + watch   │◀───────────────│  bash      │
 └────────────┘               └────────────┘   SSE events   └────────────┘
       ▲                             │
       └──── you re-run the gates ───┘
             and read the diff
```

Files here:

| File | What it is |
|---|---|
| `drive.py` | sends one brief to `POST /chat`, streams the turn, aborts a probe loop, prints an unambiguous outcome |
| `check-anchors.py` | counts each brief anchor in its file — the pre-flight that catches the commonest brief defect |
| `brief-template.md` | the brief skeleton, with every load-bearing rule already in it |
| `teeth-template.sh` | breaks the implementation on purpose to prove a new test actually fails |

---

## Before the first run

**1. The backend is running and a tool-capable model is loaded.** Coder/Instruct
variants. The Models tab marks known weak-at-tools models with ⚠️; a model that
cannot call tools reliably will narrate the edits instead of making them.

**2. Approval is settled in advance.** `drive.py` sends no
`interactive_approval`, so mutating tools fall to the configured policy: with
`approval_required = true` and no matching rule, every `edit_file` and `bash`
comes back **denied** and the turn burns its iterations arguing with itself. Add
allow rules to `~/.config/assistant/config.toml`:

```toml
[[approval_rules]]
action = "edit_file"
resource = "/Users/you/git/project-worktree/**"
decision = "allow"

[[approval_rules]]
action = "write_file"
resource = "/Users/you/git/project-worktree/**"
decision = "allow"

[[approval_rules]]
action = "bash"
decision = "allow"
```

A rule's `resource` glob matches the tool's *resource string* — a path for the
file tools, but **the command line** for `bash`. There is no path glob that
scopes bash, so allowing it is allowing arbitrary commands. Drive only a tree you
can throw away.

**3. One git worktree per task.** `POST /chat`'s `workspace` field overrides the
configured `workspace_dir` for that turn, so one backend can drive several
checkouts:

```sh
git worktree add ../wt-feature -b feature
./drive.py --brief brief.md --workspace "$(cd ../wt-feature && pwd)"
```

A worktree is also what makes a bad turn cheap: `git checkout -- .` and the
attempt never happened.

---

## The loop

### 1. Decide the change, and measure anything you are assuming

Read the code first — the real code, not your memory of it. If the change rests on
how an API behaves, prove it in a scratch program **before** writing the brief. A
brief built on a wrong assumption produces a turn that fails its gates, and the
recovery from that (below) is the most expensive thing in this whole document.

### 2. Write the brief

Copy `brief-template.md`. The parts that are not negotiable:

| Rule in the brief | The failure it prevents |
|---|---|
| every line of the new code written out verbatim, doc comments included | Asked to *write* a comment or a test, the model deliberates instead. Four runs died at the output-token cap with **zero edits made** — one spent six minutes quoting the file back to itself. Every run handed finished code placed it correctly and passed the gates first try. |
| **"Do not read any file."** | "Do not read to check something you were told" was not enough: three `read_file` calls on large sources took the prompt to 22k tokens and the turn slowed to a crawl. The brief that forbade reading outright made seven edits with no reads at all. |
| "Start with the first `edit_file` immediately; do not plan, do not decide an order" | Left to itself the model writes a multi-round plan about whether it may run edits in parallel, then re-argues it. |
| anchors of **two or more lines** | See `check-anchors.py` below — single-line anchors are the single commonest defect. |
| "if a gate fails, make one edit and re-run it; do not write probe programs" | The probe loop, below. |
| "deleting or disabling a test is a failure of this task" | It deleted a 413-line test file, and the dispatch line that ran it, rather than adapt four cases to a changed API — and reported the task complete. The instruction alone did not stop it either; the guard that works is re-running the tests yourself. |
| "report `git diff --numstat`" | A deleted suite is invisible in a green build. A file with insertions `0` is not. |
| `make fmt` is sanctioned | Otherwise a formatter-check gate fails, the model has no permission to fix it, and it starts bisecting your build system. |

State the workspace path in the brief as well as in the flag. It costs a line and
it removes the failure where a correct edit lands in the wrong tree.

### 3. Check the anchors

```sh
./check-anchors.py brief.md /Users/you/git/project-worktree
```

Every anchor must be `exact=1 loose=1`. `loose` is the count after leading
whitespace is stripped from every line, and it is the one that matters: the model
routinely re-sends an anchor with its indentation dropped, so `        checkFoo()`
— unique in the file as written — also matches `    private static func checkFoo() {`.
The edit tool refuses the ambiguous match and the model retries the identical call
until the driver kills it. Seen three times, each time on the last edit of an
otherwise finished brief.

Two things the script cannot see, so check them by eye:

- **Do not anchor on a declaration line when inserting before it.** The new code
  lands between the declaration and its own doc comment, which silently reattaches
  that comment to the wrong symbol. Anchor on the first line of the doc comment.
- **A replacement must contain the whole anchor.** When you are inserting, the
  REPLACE block is the FIND block plus the new lines, not the new lines alone.

### 4. Dispatch

```sh
./drive.py --brief brief.md --workspace ~/git/project-worktree \
    --label feature --session-file /tmp/sess-feature.json \
    --thinking off --effort low --max-iters 30 | tee feature.log
```

**`--thinking off` is the setting that makes this work.** It sets
`enable_thinking=false` in the chat template. Measured on two tracks run at once,
both with fully verbatim briefs at `--effort low`: ~20k characters of scratchpad
and **zero edits** in five minutes. The same two briefs with `--thinking off`:
first `edit_file` at 18s and 52s, no reads, no planning, six edits then the gates,
~35s per edit. Low effort shortens each thought; it does not stop the model from
having them. Turning the channel off leaves the monologue nowhere to go.

Leave thinking *on* only for a task where the model genuinely has to work
something out — which, by the rule at the top of this page, is a task it should
not have been given.

`--effort` values come from the model's own chat template; `drive.py` reads them
from `GET /models/{id}/settings` and tells you the valid list rather than letting
a bad value blow up mid-render. Both flags are absent by default, in which case
the model's configured values win.

### 5. Read the log, not the report

The transcript is one line per tool call. Read it top to bottom before you read a
word of the model's closing text, and then check the tree yourself:

```sh
git -C ../wt-feature diff --numstat     # a file with 0 insertions lost something
make fmt-check && make test             # re-run every gate the brief asked for
git -C ../wt-feature diff               # read it. all of it.
```

Treat any *explanation* of a failure in the model's report as a failure it did not
fix. One report read, in full: *"the new test suite appears to hang when run in
isolation, but this is likely due to the complexity of the initialization. The
important thing is that all existing tests continue to pass."* It had also added a
new test flag without wiring it into the target that runs the tests, so "all tests
pass" was true of the eleven suites that already existed and said nothing about
its own. Check that new tests are actually reachable from the command you run.

The last line of `drive.py`'s output is deliberately unambiguous:

```
[drive] OUTCOME: ok
[drive] OUTCOME: FAILED — 3 scratch programs in a row and no edit — …
```

A turn that ends on an error still leaves a closing paragraph above it, and read
from the bottom of a log that paragraph is indistinguishable from a report of work
done. The exit status alone was not enough either — a `| tee` in the calling shell
takes it.

### 6. Prove a new test has teeth

A test written from the same brief as the code it covers agrees with that code by
construction. It passes on the first run whether or not it checks anything. Copy
`teeth-template.sh`, write one break per fault you actually fear — the plausible
tidy-up, not a random mutation — and watch each one turn the test red:

```
drop_the_empty_string: FAILED as it should (exit 1)
swallow_the_error: *** NO TEETH *** the test still passed
restored: exit 0 (must be 0)
```

Back the files up by copy, never with `git checkout --`: it refuses paths git does
not track yet, so a brand-new file keeps every break and each later case ends up
testing the sum of the ones before it.

---

## Failure modes, and the tell for each

The three ways a turn goes wrong all look like progress from the outside. Each has
a specific tell that is cheaper than waiting.

**The probe loop.** A gate fails, the model diagnoses the cause correctly on its
first `/tmp` scratch program — and then writes six more, never touching the source
file. Still looping when killed at ~20 minutes. The stream is healthy and every
turn reads like progress, so elapsed time tells you nothing; the *tool name* is the
tell. `drive.py` aborts after three consecutive scratch-file `bash` calls with no
edit between them (`--probe-limit`, 0 disables). When it does: the diagnosis is in
the log above the abort, and the fix is usually one line. Make it by hand.

**The silent wedge.** The turn dies at the moment the model announces it will
rewrite a whole large file. The symptom is silence, not an error: the connection
stays open and the `.sse.jsonl` log stops growing. It looks exactly like a
long-running build. Check `~/.local/share/assistant/logs/backend.log` for an
`approval audit` line with **no `generation:` line after it**, and `ps` for a build
actually running in the worktree; if neither, nothing is working and it never
finishes. Kill the driver, restart the app, and re-read the file before
re-briefing — the work written so far survives, and it is usually most of the way
done. Briefing one defect at a time with the exact edit named never wedged.

**The restart that kills everything.** Restarting the backend to pick up a config
or prompt change **aborts every `/chat` turn in flight**, and the client gets an
empty body — no error, no traceback, so it reads exactly like a crash. Three tracks
once "died silently" this way and the backend got the blame; the log showed a clean
`Application is stopping` at exactly the two timestamps where the app had been
restarted by hand. Apply changes to the assistant *between* rounds, and when a turn
returns an empty body read the backend log before concluding anything.

---

## Running more than one track

Two concurrent tracks are affordable; more is usually slower in total than running
them one after another. The reason is prompt-cache eviction, not CPU: each
generation stores one prefix snapshot, and the number of snapshots kept is small,
so interleaved conversations evict each other every turn. Measured at the default
of two snapshots: prefill 8–30s and no reuse at all across nine interleaved
generations, versus 0.8–5s with a single track — about 6× worse per track.

What dominates the cost is not concurrency but **prompt growth within one turn**,
which is the real reason the brief forbids reading files: every `read_file` result
stays in the prompt for the rest of the turn, and the tail is re-prefilled each
pass.

Give each track its own worktree, its own `--label`, and its own `--session-file`.

---

## Reference

### `drive.py`

| Flag | Meaning |
|---|---|
| `--brief PATH` | the file whose contents become the message (required) |
| `--workspace PATH` | absolute path the turn's file/shell tools operate in (required) |
| `--label NAME` | names the log files; defaults to the brief's filename |
| `--base URL` | backend base URL; defaults to `$ASSISTANT_BASE_URL` or `http://127.0.0.1:9981` |
| `--model ID` | defaults to the backend's configured default model |
| `--thinking on\|off` | per-turn scratchpad switch; omit to inherit the model's setting |
| `--effort VALUE` | per-turn reasoning effort, validated against the model's template |
| `--max-iters N` | tool-call ceiling for the turn (backend clamps to 100) |
| `--session-file PATH` | continue the same conversation across runs; omit for a fresh one |
| `--out DIR` | log directory (default `./drive-logs`) |
| `--probe-limit N` | abort after N consecutive scratch-file bash calls with no edit |

Writes `DIR/LABEL.sse.jsonl` (every SSE event, the record to re-read),
`DIR/LABEL.answer.md` (the reply with the scratchpad stripped) and
`DIR/LABEL.raw.md` (the reply as streamed). Exits 0 only on a turn that reached
`done`.

### `check-anchors.py`

```sh
./check-anchors.py BRIEF [WORKSPACE]
```

Reads the brief's `FIND:` blocks, attributes each to the file named by the nearest
heading above it (`## File 1: \`path\``) or by a `Read only \`path\`` line, and
prints an exact and a whitespace-stripped occurrence count per anchor. Non-zero
exit when any anchor is not unique both ways.

### Where the numbers come from

Everything measured here was measured on an Apple Silicon Mac in August 2026,
driving `Qwen3-Coder-30B-A3B-Instruct-8bit` and `Qwen3.8-27B-8bit` through a mixed
Rust + Swift codebase. The specific timings will not transfer. The failure
*shapes* — deliberation instead of edits, dropped indentation, the probe loop, the
confident report of a failing build — have been consistent across models and are
what the rules are for.
