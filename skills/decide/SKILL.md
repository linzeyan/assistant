---
name: decide
description: Compare options and recommend one when there is a real choice to make. Use when the user asks "should I use A or B", "what's the best way to…", "which approach", or is weighing trade-offs between designs, libraries, or tools.
---

# Decide — a brief, not a shrug

When the user faces a real choice, do not waffle or list everything evenhandedly. Produce a
short decision brief that ends in a recommendation.

## 1. Frame the decision
One sentence: what is being chosen, and the single constraint that matters most here
(speed? simplicity? long-term maintainability? memory?).

## 2. Options (2–4)
For each realistic option, one line each:
- what it is
- its main advantage
- its main cost or risk

Drop options that are clearly dominated — do not pad the list.

## 3. Recommendation
Pick ONE. Say why it best fits the constraint from step 1. Name the runner-up and the one
condition under which you would switch to it.

## 4. Reversibility
Say whether this is a one-way door (hard to undo → decide carefully) or easily reversible
(→ just pick and move). Surface the call clearly; the user has the final say, so do not
bury the recommendation under "it depends".
