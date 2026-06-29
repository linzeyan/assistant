---
name: investigate
description: Find the root cause of a bug before changing any code. Use whenever something is broken, erroring, crashing, failing a test, or behaving unexpectedly — e.g. "why is this broken", "X is erroring", "it crashed", "this test fails".
---

# Investigate — root cause before fix

**Iron Law: no root cause, no fix.** Do not change code to make a symptom disappear
until you can explain *why* it happens. A patch you cannot justify usually moves the
bug, it does not remove it.

Work the stages below in order. Use your tools (read files, search code, run the
failing command, `git log`) — investigate, do not guess.

## 1. Reproduce and pin the symptom
- State the exact failure: the error message, the wrong output, or the unexpected behaviour.
- Note how to trigger it and what you expected to happen instead.
- If you can run it, run it and read the real output — do not assume what it says.

## 2. Trace the path in the code
- Find where the failure actually happens: search for the error string, or the function
  named in a traceback.
- Read that function AND its callers. Follow the data to the first point where it is wrong.
- Do not stop at the surface symptom; keep going until you reach the line truly responsible.

## 3. Check recent changes
- Run `git log -n 20 --oneline`, and `git log -p` or `git blame` on the suspect files.
- A bug that appeared "suddenly" usually rides a recent commit. Read what changed there.

## 4. Hypothesis → confirm → fix → regression test
- Write a one-sentence hypothesis of the root cause.
- Check it explains EVERY symptom from stage 1. If it does not, the hypothesis is wrong —
  go back to stage 2.
- Only then fix the cause (not the symptom).
- Add or update a test that FAILS before your fix and PASSES after, so the bug cannot return.

If you cannot find the root cause, say so and report what you ruled out. Do not apply a
guessed fix.
