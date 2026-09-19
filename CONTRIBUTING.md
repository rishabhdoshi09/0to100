# Contributing to QuantTerm

QuantTerm favors small, explicit changes over broad “improvement” patches. The goal is a system another engineer can understand and operate without tribal knowledge.

## Before changing code

Read:

1. `README.md`
2. `ARCHITECTURE.md`
3. this file
4. `OPERATIONS.md` when the change affects runtime behavior

Then identify the canonical owner of the behavior you want to change.

## Change rule

A patch should answer four questions:

1. **What requirement changes?**
2. **Which domain owns it?**
3. **What existing code becomes simpler or disappears?**
4. **How is the behavior proven?**

If a patch adds a second way to calculate the same state, schedule the same work, select the same trade, or persist the same concept, redesign it before merging.

## Prefer

- one canonical service over several wrappers;
- explicit models over loosely shaped dictionaries at domain boundaries;
- thin API routes;
- read-only UI projections;
- deterministic calculations;
- durable idempotency for scheduled/mutating work;
- deleting obsolete code after dependency proof;
- focused tests that prove behavior and architectural boundaries.

## Avoid

- speculative abstractions;
- helper files with no clear domain;
- `*_v2`, `*_new`, `*_final`, `*_parallel` as permanent architecture;
- business rules duplicated in UI, API and workers;
- hidden fallbacks that turn missing data into success;
- giant “while I am here” refactors mixed with production bug fixes.

## Definition of done

A change is complete only when:

- the canonical path uses it;
- obsolete parallel behavior is removed or explicitly quarantined;
- tests cover the intended behavior and failure mode;
- UI wording reflects backend truth;
- restart/persistence behavior is tested when stateful;
- live-money locks remain unchanged unless the task explicitly concerns certified live execution;
- root architecture/operations docs remain accurate.

## Pull requests

Keep PRs reviewable. Prefer one architectural purpose per PR. Large migrations should be staged so each step leaves the canonical product working.

The reviewer should be able to answer: “What authority changed, and why is there still only one authority after this patch?”
