# Group aggregation — consolidate raw codes into an emergent group codebook

You are consolidating open codes from ~{{GROUP_SIZE}} calls into emergent clusters.
Bottom-up: cluster by MEANING; do not force any external taxonomy.

**Domain:** {{DOMAIN_NAME}} — {{DOMAIN_DEFINITION}}

Your dispatch message lists this group's EXTRACTION FILES (JSONL), the GROUP ID, and the
OUTPUT FILE path.

## Procedure

1. Read every extraction file. Collect (a) all detections from `"gate": "coded"` lines,
   (b) counts per gate for segregated lines.
2. Cluster the detections' `candidate_category`/`candidate_sub`/`evidence_quote` by
   meaning. Aim for the natural grain: neither one mega-cluster nor a cluster per
   phrasing. Singletons that share no meaning with anything stay their own cluster.
3. For each cluster write:
   - `cluster_id`: `g{{GROUP_ID_2D}}_c<NN>` (e.g. `g00_c03`)
   - `name`: snake_case, ≤5 words, descriptive
   - `definition`: 1-2 sentences — what unites the members, boundary notes
   - `count`: number of member detections
   - `member_codes`: EVERY distinct `candidate_category` string absorbed, with its count
     — `[{"code": "...", "count": N}, ...]`. This mapping is required downstream; never
     omit or truncate it.
   - `exemplars`: 3-5 most representative `{"quote": ..., "call_id": ...}`
4. Write the OUTPUT FILE as JSON:
   `{"group": <int>, "n_calls": <int>, "n_detections": <int>, "clusters": [...],
     "segregated": {"<gate>": <count>}, "notes": ["anything surprising"]}`

## Return (this exact JSON, nothing else)

{"group": <int>, "n_clusters": <int>, "n_detections": <int>, "path": "<output file>"}
