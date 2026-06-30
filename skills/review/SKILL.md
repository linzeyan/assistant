---
name: review
description: Review a code change for correctness, scope, and tests before calling it done. Use after editing code, or when asked to "review", "check my changes", "look over this diff", or before finalising a fix.
---

# Review — check the change before declaring done

After editing code (or when asked to review a diff), do not just claim success. Read the
actual change and check it against what was asked.

## 1. Read the diff
Look at the real diff (`git diff`, or the turn's changes) — not your memory of what you
meant to write. Read every changed hunk.

## 2. Correctness
- Does it actually do what the task asked?
- Edge cases: empty input, the error path, the off-by-one, the None / nil case.
- Did it break a caller? Check who uses what you changed.

## 3. Scope
- Anything changed that the task did NOT require — stray reformatting, unrelated "fixes"?
  Flag it or revert it. Keep the change surgical.

## 4. Tests
- Is there a test that fails without this change and passes with it? If the behaviour is
  new or changed and untested, add the test or name the one to add.

## 5. Verdict
State it plainly: done and verified, or exactly what is still open. "Tests pass" is only
true if you ran them — if you did not, say so rather than implying it.
