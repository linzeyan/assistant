#!/usr/bin/env bash
# Prove a new test has teeth: break the implementation on purpose, one fault at
# a time, and check the test actually fails.
#
# A test written from the same brief as the code it covers agrees with that code
# by construction — it passes on the first run whether or not it is checking
# anything. The only evidence that it checks something is watching it go red for
# a reason you chose. Copy this file per feature, fill in the two config blocks,
# and read the summary at the end: every break must FAIL, and the restored tree
# must pass.
#
# Deliberately no `set -e`: a break that fails to build is information, and the
# script must still restore the tree afterwards.

WORKSPACE=${WORKSPACE:-$PWD}
OUT=${OUT:-$WORKSPACE/teeth-logs}
BUILD=${BUILD:-make build}
VERIFY=${VERIFY:-make test}

# ---- config 1: the files any break below edits ------------------------------
# Backed up by copy, not by `git checkout --`, which refuses paths git does not
# track yet — a brand-new file would then keep every break, and each later case
# would be testing the sum of the ones before it.
FILES=(
  src/thing.ext
)

# ---- config 2: one function per fault ---------------------------------------
# Each is a plausible tidy-up a later reader might make, not a random mutation:
# the question the test has to answer is "would this change be caught", and a
# nonsense edit answers nothing. `replace_once` asserts the text it replaces
# appears exactly once, so a break that has drifted out of date is loud.
BREAKS=(drop_the_empty_string swallow_the_error)

break_drop_the_empty_string() {
  replace_once src/thing.ext \
    'password: key.isEmpty ? secret : nil' \
    'password: key.isEmpty && !secret.isEmpty ? secret : nil'
}

break_swallow_the_error() {
  replace_once src/thing.ext 'return err' 'return nil'
}

# ---- runner ------------------------------------------------------------------

replace_once() {
  /usr/bin/env python3 - "$WORKSPACE/$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
body = open(path, encoding="utf-8").read()
count = body.count(old)
assert count == 1, f"{path}: anchor appears {count} times, expected 1"
open(path, "w", encoding="utf-8").write(body.replace(old, new))
PY
}

rm -rf "$OUT"; mkdir -p "$OUT"
cd "$WORKSPACE" || exit 1

for f in "${FILES[@]}"; do
  mkdir -p "$OUT/orig/$(dirname "$f")"
  cp "$f" "$OUT/orig/$f" || exit 1
done
restore() { for f in "${FILES[@]}"; do cp "$OUT/orig/$f" "$f"; done; }
trap restore EXIT

summary="$OUT/summary.txt"
: >"$summary"

for name in "${BREAKS[@]}"; do
  restore
  if ! "break_$name"; then
    echo "$name: could not apply the break — the anchor has moved" | tee -a "$summary"
    continue
  fi
  if ! $BUILD >"$OUT/$name-build.log" 2>&1; then
    # A break that does not compile proves nothing about the test: the compiler
    # caught it, and the test never ran. Pick a fault that builds.
    echo "$name: BUILD FAILED — see $OUT/$name-build.log" | tee -a "$summary"
    continue
  fi
  $VERIFY >"$OUT/$name-verify.log" 2>&1
  code=$?
  if [ $code -ne 0 ]; then
    echo "$name: FAILED as it should (exit $code)" | tee -a "$summary"
  else
    echo "$name: *** NO TEETH *** the test still passed" | tee -a "$summary"
  fi
done

restore
$BUILD >"$OUT/restored-build.log" 2>&1
$VERIFY >"$OUT/restored-verify.log" 2>&1
echo "restored: exit $? (must be 0)" | tee -a "$summary"
