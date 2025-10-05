# ADR 0001: Adopt Architecture Decision Records

## Status
Accepted

## Context
The project has grown to include multiple interacting packages and components. Without a shared process for capturing
design rationale, contributors risk duplicating past discussions or introducing changes that conflict with existing
architecture decisions.

## Decision
We will maintain Architecture Decision Records (ADRs) under `docs/adr/`. Each ADR will document the problem being
addressed, the selected solution, the rationale behind that solution, and any implications for future work. Any change
that affects application architecture, cross-cutting concerns, or long-lived design choices must reference an ADR.

## Consequences
- Contributors must review existing ADRs before proposing architectural changes or implementing features that interact
  with the affected areas.
- New architectural decisions require a new ADR that follows this template.
- Pull requests introducing architectural changes must link to the relevant ADR(s) in their description.
- The AGENTS instructions reference the ADR process so that future agents stay aligned with the documented design.

## Implementation Notes
- Store ADR files using the naming convention `NNNN-title.md` where `NNNN` is a zero-padded sequence number.
- Keep ADRs in chronological order and avoid renaming files to preserve history.
- Use Markdown headings: Status, Context, Decision, Consequences, and Implementation Notes.
