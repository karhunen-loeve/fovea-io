#!/bin/sh
#
# Released CHANGELOG sections are immutable.
#
# For every `vX.Y.Z` tag, the `## [X.Y.Z]` section in the working tree's
# CHANGELOG.md must byte-match the same section as it was at that tag.
# A release is a promise about what shipped; docs.rs serves the changelog
# to users, so silently rewriting a shipped entry misrepresents a published
# version.
#
# This catches two real failure modes, both of which have happened:
#
#   1. A deleted version heading. The release commit that cuts
#      `[Unreleased]` into `[X.Y.Z]` lives only on `main`, so `develop`
#      carries its own copy of that heading and nothing else guards it.
#      A later edit that appends a `### Added` block can delete the
#      heading with it, folding every released entry back into
#      `[Unreleased]` where it silently looks unpublished.
#
#   2. A retroactive edit to a shipped entry, rewriting it to describe
#      current behaviour instead of what that version did. This also hides
#      the change itself, which never gets listed under `[Unreleased]`.
#
# Usage: check-changelog-immutable.sh [candidate file]
#
# The tag side is always `CHANGELOG.md` at the repository root — that is the
# published path. The optional argument overrides only the *candidate* file
# to compare against it, which is how this script's own test fixtures work.
# Exits 0 if every released section is intact, 1 otherwise.
#
# Requires the full tag history: in CI, check out with `fetch-depth: 0`.

set -eu

# Path inside the tagged trees. Not configurable: a release published this
# exact path, and making it an argument would let a typo turn every
# comparison into a silent skip.
readonly RELEASED_PATH="CHANGELOG.md"

candidate="${1:-$RELEASED_PATH}"

if [ ! -f "$candidate" ]; then
    echo "error: $candidate not found (run from the repository root)" >&2
    exit 1
fi

# Extract the `## [<version>]` section, heading line included, up to but
# excluding the next `## [` heading or the link-reference block at the end.
# Bare `##` sub-headings (`### Added`) are left alone: the pattern anchors
# on `## [`, and awk's `index` check keeps `###` from matching.
extract_section() {
    awk -v want="## [$2]" '
        # A section heading is `## [` at the start of a line.
        /^## \[/ {
            if (inside) exit
            if (index($0, want) == 1) { inside = 1; print; next }
        }
        # The trailing `[X.Y.Z]: https://…` compare-link block ends the
        # last section.
        inside && /^\[[0-9]/ { exit }
        inside { print }
    ' "$1"
}

tags=$(git tag --list 'v[0-9]*.[0-9]*.[0-9]*' | sort -V)

if [ -z "$tags" ]; then
    echo "error: no vX.Y.Z tags found; is this a shallow clone?" >&2
    echo "       CI must check out with fetch-depth: 0" >&2
    exit 1
fi

status=0
checked=0

for tag in $tags; do
    version=${tag#v}

    # The changelog may not have existed, or may not have carried a section
    # for this version, at the tag itself. Nothing to be immutable about.
    if ! git cat-file -e "$tag:$RELEASED_PATH" 2>/dev/null; then
        echo "skip  $tag — no $RELEASED_PATH at this tag"
        continue
    fi

    released=$(git show "$tag:$RELEASED_PATH" | extract_section /dev/stdin "$version" || true)
    if [ -z "$released" ]; then
        echo "skip  $tag — no [$version] section at this tag"
        continue
    fi

    current=$(extract_section "$candidate" "$version" || true)
    if [ -z "$current" ]; then
        echo "FAIL  $tag — the [$version] section is missing from $candidate" >&2
        echo "      It shipped at $tag but no '## [$version]' heading remains." >&2
        echo "      A released heading was most likely deleted by a later edit;" >&2
        echo "      restore it (its entries are probably now inside" >&2
        echo "      [Unreleased]) rather than re-deriving the text." >&2
        status=1
        continue
    fi

    if [ "$released" = "$current" ]; then
        echo "ok    $tag — [$version] section matches the release"
    else
        echo "FAIL  $tag — the [$version] section was modified after release" >&2
        echo "      Released sections describe a published version and must not" >&2
        echo "      change. Put the new behaviour under [Unreleased] instead." >&2
        echo "      Diff (< as released at $tag, > as in the working tree):" >&2
        printf '%s\n' "$released" > "${TMPDIR:-/tmp}/changelog-released.$$"
        printf '%s\n' "$current" > "${TMPDIR:-/tmp}/changelog-current.$$"
        diff "${TMPDIR:-/tmp}/changelog-released.$$" \
             "${TMPDIR:-/tmp}/changelog-current.$$" >&2 || true
        rm -f "${TMPDIR:-/tmp}/changelog-released.$$" \
              "${TMPDIR:-/tmp}/changelog-current.$$"
        status=1
    fi

    checked=$((checked + 1))
done

echo
if [ "$status" -eq 0 ]; then
    echo "All $checked released CHANGELOG section(s) intact."
else
    echo "Released CHANGELOG section(s) differ from what was published." >&2
fi

exit "$status"
