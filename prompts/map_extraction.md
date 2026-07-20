# Map extraction — open-code one chunk of calls

You are open-coding call-center transcriptions for bottom-up discovery. Do not use any
predefined taxonomy: name what you see, in the data's own terms.

**Domain:** {{DOMAIN_NAME}} — {{DOMAIN_DEFINITION}}
**Unit of analysis:** {{UNIT}}
**Language notes:** {{LANGUAGE_NOTES}}

Your dispatch message tells you the CHUNK FILE (input), the OUTPUT FILE, and the EXPECTED
number of calls.

## Procedure

1. Read the chunk file. Each line is `{"call_id": ..., "transcription": ...}`.
2. For EVERY call — no exceptions, even garbage — write exactly ONE JSON line to the
   output file (create it; if you were dispatched to patch specific missing call ids,
   APPEND only lines for those ids):
   `{"call_id": "...", "gate": "...", "detections": [...]}`
3. **Gate first.** Before coding, classify the call:
{{GATES}}
   If a gate applies, set `"gate"` to that gate's name and `"detections": []`. Otherwise
   `"gate": "coded"`.
4. **Code coded calls.** A detection is an instance of the domain concept:
   - `candidate_category`: a short free-form label (3-6 words, lowercase) in YOUR words,
     specific enough that two different problems never share a label.
   - `candidate_sub`: optional finer label, else `""`.
   - `evidence_quote`: {{EVIDENCE_RULE}}
   - Unit rule: {{UNIT_RULE}}
   A coded call with no instance of the concept gets `"detections": []` — that is a
   valid, common outcome. Never invent detections.
5. Self-check: count your output lines vs EXPECTED. Report any call ids you could not
   emit in `missing`.

## Return (this exact JSON, nothing else)

{"chunk": "<chunk file name>", "expected": <int>, "written": <int>, "missing": ["<call_id>", ...]}
