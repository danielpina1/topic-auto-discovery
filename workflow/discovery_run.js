export const meta = {
  name: 'topic-discovery-run',
  description: 'Map (sonnet) + per-group aggregation (opus) for one topic-discovery run',
  whenToUse: 'Dispatched by the topic-auto-discovery skill; not for direct use.',
  phases: [
    { title: 'Map', detail: 'sonnet open-coding, one agent per 20-call chunk', model: 'sonnet' },
    { title: 'Aggregate', detail: 'opus emergent codebook per group', model: 'opus' },
  ],
}

const MAP_SCHEMA = {
  type: 'object',
  properties: {
    chunk: { type: 'string' },
    expected: { type: 'integer' },
    written: { type: 'integer' },
    missing: { type: 'array', items: { type: 'string' } },
  },
  required: ['chunk', 'expected', 'written', 'missing'],
}

const AGG_SCHEMA = {
  type: 'object',
  properties: {
    group: { type: 'integer' },
    n_clusters: { type: 'integer' },
    n_detections: { type: 'integer' },
    path: { type: 'string' },
  },
  required: ['group', 'n_clusters', 'n_detections', 'path'],
}

const cfg = typeof args === 'string' ? JSON.parse(args) : args
const { runDir, mapPrompt, aggPrompt, groups } = cfg

const outName = (f) => f.replace('.jsonl', '.out.jsonl')

const mapChunk = (c, extra) =>
  agent(
    `Follow the instructions in ${mapPrompt} EXACTLY.\n` +
      `CHUNK FILE: ${runDir}/chunks/${c.file}\n` +
      `OUTPUT FILE: ${runDir}/extractions/${outName(c.file)}\n` +
      `EXPECTED: ${c.n}\n` + (extra || ''),
    { label: `map:${c.file}`, phase: 'Map', model: 'sonnet', effort: 'low', schema: MAP_SCHEMA },
  )

const results = await pipeline(
  groups,
  async (g) => {
    const outs = (await parallel(g.chunks.map((c) => () => mapChunk(c)))).filter(Boolean)
    const short = outs.filter((o) => o.missing && o.missing.length > 0)
    for (const s of short) {
      const c = g.chunks.find((x) => x.file === s.chunk) || { file: s.chunk, n: s.expected }
      log(`re-dispatching ${s.chunk}: ${s.missing.length} missing`)
      await mapChunk(c, `PATCH MODE: the output file exists; APPEND lines ONLY for these missing call ids: ${s.missing.join(', ')}\n`)
    }
    return g
  },
  async (g) => {
    const gid = String(g.id).padStart(2, '0')
    const files = g.chunks.map((c) => `${runDir}/extractions/${outName(c.file)}`).join('\n')
    return agent(
      `Follow the instructions in ${aggPrompt} EXACTLY.\n` +
        `GROUP ID: ${g.id}\n` +
        `EXTRACTION FILES:\n${files}\n` +
        `OUTPUT FILE: ${runDir}/groups/group_${gid}.json\n`,
      { label: `aggregate:g${gid}`, phase: 'Aggregate', model: 'opus', schema: AGG_SCHEMA },
    )
  },
)

return { groups: results.filter(Boolean) }
