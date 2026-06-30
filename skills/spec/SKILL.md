---
name: spec
description: Turn a vague or broad request into a short focused spec before writing code. Use when the ask is fuzzy, large, or open-ended ("build X", "add a feature for Y", "I want something that…") and the scope isn't pinned down yet.
---

# Spec — focus before building

When a request is vague, broad, or open-ended, do NOT start coding. First write a short
spec so you build the right thing. Keep each section to a few lines, not an essay.

## 1. Goal (one sentence)
What outcome does the user actually want? State it in a single sentence. If you cannot,
the request is still ambiguous — ask ONE sharp clarifying question and stop here.

## 2. Scope / non-goals
- In scope: the smallest set of changes that delivers the goal.
- Out of scope: tempting extras you will NOT do this round. Name them so they are parked,
  not silently dropped.

## 3. Approach
- The files / modules you expect to touch — read them first to confirm the shape.
- The one main design decision, and the option you would pick, with a one-line why.

## 4. Acceptance
- How both of you will know it is done: the command to run, the test that should pass,
  the behaviour to observe.

Show this spec, get a nod (or adjustment), THEN implement. For a small, unambiguous task,
skip the ceremony and just do it — this skill is for when the scope is unclear.
