<!--
Copy this file, fill in the placeholders, delete these comments, then:

    ./check-anchors.py my-brief.md /path/to/workspace
    ./drive.py --brief my-brief.md --workspace /path/to/workspace \
        --thinking off --effort low

Everything below the placeholders is load-bearing — each rule is there because a
run failed without it. See README.md for which failure each one prevents.
-->

# Task: <one line, in the imperative — what the tree should look like afterwards>

Workspace: `<absolute path>` (this is the directory you are in).

Apply the edits below **exactly as written**. Do not design anything, do not
improve anything, do not rename anything. Every anchor below appears exactly
once in its file. Match on the text ignoring leading whitespace, and keep the
indentation shown.

**Do not read any file.** Everything you need is in this brief. There are
<N> edits across <M> files.

Start with the first `edit_file` call immediately: do not plan, do not decide an
order, do not consider whether the edits can be made in parallel.

`<make fmt>` is sanctioned and expected: run it after the edits.

---

## File 1: `<path/relative/to/the/workspace>`

### Edit 1 — <why this edit exists, one line>

FIND:
```<language>
<two or more lines, copied out of the file exactly, including the indentation>
```

REPLACE WITH:
```<language>
<the finished text, every line written out — no placeholders, no "…as above">
```

### Edit 2 — <why>

FIND:
```<language>
<...>
```

REPLACE WITH:
```<language>
<...>
```

---

## File 2: `<path>`

### Edit 3 — <why>

FIND:
```<language>
<...>
```

REPLACE WITH:
```<language>
<...>
```

---

## Then

1. Run `<make fmt>`.
2. Run `<the test command>` and report the output verbatim.
3. Run `<the format/lint check>` and report the output verbatim.
4. Report `git diff --numstat`.

If any of them fails, report the failure verbatim and stop. Make one edit and
re-run the gate; do not write probe programs, do not attempt a different design,
and do not edit anything not listed above. Deleting or disabling a test is a
failure of this task, not a way to make it pass.
