# Discovery report — write the cumulative domain report

You are the reporter node. Write `reports/DISCOVERY_REPORT.md` for a reader who has
NEVER seen this project: a complete, self-contained account of the auto-discovery — its
assumptions, its taxonomy, and its evidence. Overwrite the previous version (git keeps
history). Your dispatch message gives the workspace path; read `domain.md`,
`codebook.json`, and the latest run's `umap/clusters.json` before writing.

Image embeds use relative paths from `reports/`: `![...](assets/scatter_categories.png)`.

## Required structure

1. **Executive summary** — the domain in one paragraph; taxonomy size and stability;
   headline findings; saturation verdict.
2. **Methodology & assumptions** — make every assumption explicit:
   - domain definition quoted verbatim from `domain.md`;
   - sampling design: table, window, FARM_FINGERPRINT determinism, sample size/offset
     per run (from the ledger);
   - unit of analysis and what it implies; scope gates and what they exclude (with
     segregated volumes); evidence rule;
   - pipeline: sonnet-5 open-coding → opus-4.8 group aggregation → fable-5 collector
     with persistent memory; UMAP/BGE-M3 cross-check lane (advisory);
   - taxonomy caps; known limitations and biases (sample ≠ population, LLM coder
     variance, embedding-model behavior on the corpus languages).
3. **The taxonomy** — for EVERY non-tombstone category: id, label, status, full
   definition WITH boundary description (what's in, what's out, adjacent categories),
   lifecycle (first/last seen run), absorbed aliases, total + per-run counts, 2-3
   verbatim exemplar quotes with call ids. Order by total count, grouped by family when
   two_level. Then a short Tombstones subsection (what merged into what, why).
4. **UMAP evidence** — embed `assets/scatter_categories.png`, `assets/families.png`
   (when present) and `assets/agreement_matrix.png`; link `assets/scatter.html`.
   Interpret: where geometry and taxonomy agree; every over-split / scatter /
   missed-topic signal and what the collector decided about it.
5. **Cross-run picture** — runs ledger table (run, date, calls, new, merged, confirmed,
   saturated); category-arrival timeline; saturation trajectory and expected next-run
   yield; `open_questions` from the codebook.
6. **Appendix** — file inventory of the workspace and how to reproduce a run.

Style: plain prose, no hedging, every claim traceable to codebook fields or assets.

## Return (this exact JSON, nothing else)

{"path": "reports/DISCOVERY_REPORT.md", "categories_documented": <int>, "images_embedded": <int>}
