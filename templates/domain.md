# Domain Definition — {{CLIENT}} / {{DOMAIN_NAME}}

Approved by user on: {{APPROVED_DATE}}

## Concept
**Name:** {{DOMAIN_NAME}}
**Working definition:** {{DOMAIN_DEFINITION}}

## Unit of analysis
{{UNIT}}
<!-- "call" (at most one detection per call) or "event" (a call can carry several detections) -->

## Scope gates (segregate — record but never code)
{{GATES}}
<!-- Default-on, one per line, format: gate_name — description. Baseline:
agent_internal — agent-to-agent / internal coordination, no customer on the line
unintelligible — noise, dead air, non-call audio, or transcript too corrupted to code -->

## Evidence rule
{{EVIDENCE_RULE}}
<!-- Default: every detection MUST carry a verbatim quote from the transcription, in its original language. -->

## Taxonomy shape
**Shape:** {{TAXONOMY_SHAPE}}   <!-- flat | two_level -->
**Caps:** {{CAPS}}              <!-- e.g. two_level: max 8 families, max 6 subs each. flat: max 25 categories -->

## Data source
- **Table:** `{{TABLE}}`
- **Call id column:** `{{ID_COL}}`
- **Transcription column:** `{{TEXT_COL}}`
- **Date column:** `{{DATE_COL}}` (blank if none)
- **Window:** {{WINDOW}}
- **Extra WHERE:** `{{WHERE}}` (blank if none)

## Run sizing
- **Sample size:** {{SAMPLE_SIZE}} (deterministic FARM_FINGERPRINT order; use --offset on later runs for fresh calls)
- **Chunk size:** 20 calls/agent · **Group size:** 1000 calls/group

## Language notes
{{LANGUAGE_NOTES}}
