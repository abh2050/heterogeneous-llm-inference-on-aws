'use strict';
// Guided one-page experience. Shares `state`, `api`, `plan`, `recalculate`, `hw`, `esc`, `fmt`, `money`
// with the Advanced section (app.js). All numbers come from the same server calculations.

const G = {job: 'chat', totalUsers: 100, activePct: 10, precision: 'fp8', apiPrice: 10, hardwareId: null, layer: 'users', timer: null, seq: 0};
const READING_TPS = 20, WAIT_OK_S = 2, HOURS_PER_MONTH = 730;

const PRECISIONS = {
  bf16: {name: 'Best quality', code: 'BF16', bytes: 2, line: 'Full-precision numbers. The most accurate answers, and the most memory.'},
  fp8: {name: 'Balanced', code: 'FP8', bytes: 1, line: 'Half the memory of Best quality with almost no visible difference. A good default.'},
  int4: {name: 'Smallest', code: 'INT4', bytes: 0.5, line: 'A quarter of the memory. Fits on modest hardware, with a small loss in quality.'},
};
const JOB_ICONS = {chat: '💬', rag: '📄', coding: '⌨️', agents: '🤖'};
const JOB_LINES = {
  chat: 'A back-and-forth conversation. Short questions, so the speed of each reply matters most.',
  rag: 'Answers questions about your documents. Each question carries pages of text, so the wait before the first word grows.',
  coding: 'Helps write and explain code. Long files in, long answers out, so it needs both memory and speed.',
  agents: 'Software that plans and calls tools in loops. Agents send long prompts again and again, so reusing cached prompts matters most.',
};

const GLOSSARY = {
  token: ['Token', 'A small piece of text, roughly three quarters of a word, that the model reads or writes one at a time.'],
  model: ['Model', 'The trained AI program, stored as a very large file of numbers.'],
  parameters: ['Parameters', 'The numbers inside a model; “30B” means 30 billion of them, and more usually means smarter and bigger.'],
  gpu: ['GPU', 'A chip built to do huge amounts of math in parallel, the usual engine for AI.'],
  memory: ['Memory (VRAM)', 'The fast storage attached to the chip; the model and its sticky notes must fit in it.'],
  bandwidth: ['Memory bandwidth', 'How fast the chip can pull data from its memory, which mostly sets how fast words appear.'],
  ttft: ['TTFT (time to first token)', 'The wait between sending a question and seeing the first word of the answer.'],
  tps: ['Tokens per second', 'How many tokens one person sees per second; around 20 feels comfortable to read.'],
  prefill: ['Prefill', 'The step where the model reads your whole question before it starts answering.'],
  decode: ['Decode', 'The step where the model writes the answer one token at a time.'],
  kv: ['KV cache', 'The model’s sticky notes about the conversation so far, kept in memory so it never rereads everything.'],
  context: ['Context length', 'How much text (question plus answer plus history) the model can hold in mind at once.'],
  precision: ['Precision', 'How many bits each model number uses; fewer bits save memory at a small cost in accuracy.'],
  bf16: ['BF16', 'A 16-bit number format (2 bytes per number) that keeps full quality.'],
  fp8: ['FP8', 'An 8-bit number format (1 byte per number), half the memory of BF16.'],
  int4: ['INT4', 'A 4-bit format (half a byte per number), a quarter of the memory of BF16.'],
  quantization: ['Quantization', 'Storing the model’s numbers with fewer bits to save memory.'],
  moe: ['Mixture of experts', 'A model made of many specialist parts where only a few work on each word, so it is big in memory but quick to run.'],
  batching: ['Batching', 'Serving many people with one pass over the model so the expensive memory trip is shared.'],
  concurrent: ['Concurrent users', 'People whose requests are in flight at the same moment.'],
  engine: ['Serving engine', 'The program that loads the model and answers requests, like a kitchen that cooks the orders.'],
  scheduler: ['Scheduler', 'The traffic director that decides which server handles each request, like a restaurant host.'],
  gateway: ['Gateway', 'The front door that receives requests from your apps and passes them on.'],
  vllm: ['vLLM', 'A popular open-source serving engine built for many users at once.'],
  sglang: ['SGLang', 'Another open-source serving engine, strong at reusing shared prompt prefixes.'],
  llamacpp: ['llama.cpp', 'A lightweight engine that runs models on laptops, small boxes, and CPUs.'],
  ollama: ['Ollama', 'An easy desktop tool, built on llama.cpp, for running models locally with one command.'],
  llmd: ['llm-d', 'An open-source Kubernetes layer that spreads vLLM across many servers and routes requests smartly.'],
  kubernetes: ['Kubernetes', 'Software that runs and manages many servers as one system.'],
  sharding: ['Sharding (tensor parallelism)', 'Splitting one model across several GPUs when it is too big for one.'],
  nvlink: ['NVLink', 'A very fast link between NVIDIA GPUs, so shards can swap data without waiting.'],
  disaggregated: ['Disaggregated serving', 'Running prefill and decode on separate machines and passing the sticky notes between them.'],
  heterogeneous: ['Heterogeneous serving', 'Using different kinds of chips for different jobs, such as one for prefill and another for decode.'],
  replica: ['Replica', 'One complete copy of the model serving requests; more replicas handle more people.'],
  cuda: ['CUDA', 'NVIDIA’s software layer that lets programs use its GPUs.'],
  rocm: ['ROCm', 'AMD’s software layer that lets programs use its GPUs.'],
  oneapi: ['oneAPI', 'Intel’s software layer that lets programs use its GPUs and CPUs.'],
};
function T(key, text) {const g = GLOSSARY[key]; return `<a class="term" href="#g-${key}" data-term="${key}" title="${esc(g[1])} (see glossary)">${text ?? g[0]}</a>`;}

// ---- labels: every number gets measured | calculated | estimated plus an explanation
function lab(value, kind, how) {return `<span class="g-num">${value}</span><span class="tag tag-${kind}">${kind}</span><button type="button" class="info" data-tip="${esc(how)}" aria-label="How this was ${kind}: ${esc(how)}">i</button>`;}
function evidenceKind(c) {return c.evidence === 'Measured' ? 'measured' : 'calculated';}

// ---- derived guided inputs
function peakUsers() {return Math.max(1, Math.round(G.totalUsers * G.activePct / 100));}
function precisionLabel() {return PRECISIONS[G.precision];}

async function applyGuided() {
  const seq = ++G.seq, job = state.presets.find(p => p.id === G.job), peak = peakUsers();
  const w = {...state.workload, ...job.workload, total_users: G.totalUsers, peak_concurrent_users: peak};
  // Little's law: each active person has one request in flight; a reply lasts about output / 20 tok/s + 1 s.
  w.peak_rps = Math.max(0.01, peak / (w.output_avg / READING_TPS + 1));
  state.workload = w;
  const bytes = PRECISIONS[G.precision].bytes;
  state.assumptions = {...state.assumptions, weight_precision: G.precision, weight_bytes: bytes};
  try {await recalculate();} catch (e) {toast(e.message, true); return;}
  if (seq !== G.seq) return;
  renderResults(); renderStack(); renderModelCard();
  if (document.getElementById('advanced').open) render();
}
function schedule() {clearTimeout(G.timer); G.timer = setTimeout(applyGuided, 250);}

// ---- hardware options that fit
function mark(c) {
  const tps = 1 / c.tpot_service_s, wait = c.ttft_service_s;
  if (tps >= READING_TPS && wait <= WAIT_OK_S) return 'green';
  if (tps < 10 || wait > 5) return 'red';
  return 'yellow';
}
const RANK = {green: 0, yellow: 1, red: 2};
function options() {
  const need = state.workload.peak_concurrent_users, best = {};
  for (const c of state.candidates) {
    if (c.tpot_service_s == null || c.ttft_service_s == null || !c.replicas || !c.hourly_usd) continue;
    if (!(c.concurrent_users_capacity >= need)) continue;
    const m = mark(c), cur = best[c.hardware_id];
    if (!cur || RANK[m] < RANK[cur.mark] || (RANK[m] === RANK[cur.mark] && c.hourly_usd < cur.c.hourly_usd)) best[c.hardware_id] = {c, mark: m};
  }
  return Object.values(best).sort((a, b) => a.c.hourly_usd - b.c.hourly_usd);
}
function chosen(opts) {
  return opts.find(o => o.c.hardware_id === G.hardwareId) || opts.find(o => o.mark === 'green') || opts[0];
}
function speedWord(tps) {
  if (tps < 10) return 'laggy, close to reading speed';
  if (tps < 20) return 'faster than reading, but a little slow for long answers';
  if (tps < 30) return 'comfortable, faster than reading';
  if (tps < 50) return 'fast';
  return 'instant';
}
function waitWord(s) {return s < 1 ? 'instant' : s < 2 ? 'fine' : s <= 5 ? 'noticeable' : 'too slow';}

// ---- Step 1 model card
function renderModelCard() {
  const m = state.model, el = document.getElementById('g-model-card');
  if (!el || !m) return;
  const p = m.parameters, active = m.active_parameters || p, moe = !!(m.num_experts && m.num_experts_per_tok);
  const mem = (b) => p ? lab(fmt(p / 1e9 * b, 1) + ' GB', 'calculated', `Parameters (${fmt(p / 1e9, 1)} billion) × ${b} bytes per number. Weights only; conversations need extra memory.`) : 'unknown';
  el.innerHTML = `<div class="g-card"><h3>${esc(m.model_id)}</h3>
    <p>Size: ${p ? lab(fmt(p / 1e9, 1) + ' billion parameters', 'calculated', 'Read from the model’s published files (or its config when the file sizes are not listed).') : 'not published'} </p>
    <p>Memory needed just to hold it: ${T('bf16', 'BF16')} ${mem(2)} · ${T('fp8', 'FP8')} ${mem(1)} · ${T('int4', 'INT4')} ${mem(0.5)}</p>
    <p>${T('moe', 'Mixture of experts')}: <b>${moe ? 'yes' : 'no'}</b>. ${moe
      ? `It has ${m.num_experts} specialist parts but uses only ${m.num_experts_per_tok} for each word (about ${fmt(active / 1e9, 1)} billion active), so it needs memory for all of them yet answers like a much smaller model.`
      : 'Every part of the model works on every word, so memory and speed both scale with its size.'}</p>
    ${m.revision ? '' : '<p class="g-note">This is an offline example. Search above to look up a real model.</p>'}
    ${m.unsupported?.length ? `<p class="g-note">Not fully supported yet: ${esc(m.unsupported.join('; '))}</p>` : ''}</div>`;
}

// ---- Results
function renderResults() {
  const el = document.getElementById('g-results'); if (!el) return;
  const opts = options(), prec = precisionLabel(), peak = state.workload.peak_concurrent_users;
  const missing = (state.hardware_catalog || []).filter(r => !r.usable);
  const sparkPrompt = G.hardwareId && missing.find(r => r.id === G.hardwareId);
  let html = `<h2>Your result</h2>`;
  if (sparkPrompt) html += missingForm(sparkPrompt);
  const o = chosen(opts);
  if (!o) {
    el.innerHTML = html + `<p class="g-verdict">Nothing in the catalog fits this job at ${prec.name} quality for about ${fmt(peak, 0)} people at once.</p><p class="g-sub">Try fewer people, the Smallest choice, or a lighter job. ${missing.length ? `${missing.length} catalog entries have unpublished values and are not included; you can fill them in under Advanced.` : ''}</p>`;
    return;
  }
  const c = o.c, h = hw(c.hardware_id), tps = 1 / c.tpot_service_s, kind = evidenceKind(c);
  const mem = c.memory, other = mem.runtime + mem.activations + mem.communication + mem.safety;
  const physical = mem.physical, free = Math.max(0, physical - mem.weights - mem.kv - other);
  const pct = (x) => (100 * x / physical).toFixed(1) + '%';
  const gaugeMax = 60, at = (v) => Math.min(100, 100 * v / gaugeMax).toFixed(1) + '%';
  const monthly = c.hourly_usd * HOURS_PER_MONTH, perM = c.cost_per_million_output;
  const api = Number(G.apiPrice) || 0, cheaper = perM != null && api > 0 ? (1 - perM / api) * 100 : null;
  const where = `${c.device_count} ${esc(h.name)}`;
  html += `<p class="g-verdict" aria-live="polite">${o.mark === 'green' ? `This fits on ${where}.` : `The closest option is ${where}, and it misses a speed or wait target.`} Each person gets about ${fmt(tps, 0)} tokens per second, which is ${speedWord(tps)}.</p>
  <div class="g-grid">
    <div class="g-card"><h3>Memory on each device</h3><div class="g-bar" role="img" aria-label="Memory split">
      <span class="w" style="width:${pct(mem.weights)}"></span><span class="k" style="width:${pct(mem.kv)}"></span><span class="r" style="width:${pct(other)}"></span><span class="f" style="width:${pct(free)}"></span></div>
      <div class="g-legend"><span><i style="background:#0c796c"></i>Model weights ${lab(fmt(mem.weights / 1e9, 1) + ' GB', 'calculated', 'Parameters × bytes per number, plus a small format overhead, divided across the devices.')}</span>
      <span><i style="background:#7b64a4"></i>Conversation memory (${T('kv', 'KV cache')}) ${lab(fmt(mem.kv / 1e9, 1) + ' GB', 'calculated', '2 × layers × KV heads × head size × bytes × tokens per conversation × conversations.')}</span>
      <span><i style="background:#c2cbce"></i>System reserve ${fmt(other / 1e9, 1)} GB</span>
      <span><i style="background:#e7f4ee;border:1px solid #cde"></i>Free ${lab(fmt(free / 1e9, 1) + ' GB', 'calculated', 'The device’s memory minus everything above.')}</span></div>
      <p class="g-note">of ${fmt(physical / 1e9, 0)} GB on each ${esc(h.name)}${c.tp > 1 ? ` (${c.tp} share the model)` : ''}.</p></div>
    <div class="g-card"><h3>Speed for each person</h3><div class="g-value">${lab(fmt(tps, 0) + ' tokens/s', kind, 'One divided by the time the chip needs for each word at this many people. Calculated from memory bandwidth and compute; a real benchmark replaces it.')}</div>
      <div class="g-gauge" role="img" aria-label="${fmt(tps, 0)} tokens per second; comfortable at 20"><span class="you" style="left:${at(tps)}">${fmt(tps, 0)}</span><span class="pin" style="left:${at(READING_TPS)}"></span>
      <span class="mark" style="left:${at(10)}">10 laggy</span><span class="mark" style="left:${at(25)}">20–30 comfortable</span><span class="mark" style="left:${at(55)}">50+ instant</span></div>
      <p class="g-note">The dark line marks 20 tokens per second. ${T('tps', 'What is a token per second?')}</p></div>
    <div class="g-card"><h3>Wait before the first word (${T('ttft', 'TTFT')})</h3><div class="g-value">${lab(fmt(c.ttft_service_s, 2) + ' s', kind, 'Time to read the whole question (prefill) plus the first step of the answer. Calculated; queueing is not included.')}</div>
      <p class="g-note">${waitWord(c.ttft_service_s)}. Instant under 1 second, fine under 2, too slow over 5.</p></div>
    <div class="g-card"><h3>Cost</h3><div class="g-value">${lab(money(monthly, 0) + ' / month', 'calculated', `Hourly price of ${c.device_count} device(s) × ${HOURS_PER_MONTH} hours.`)}</div>
      <p>${lab(money(perM, 2) + ' per million tokens', 'calculated', 'Hourly cost ÷ (words actually served per second × 3600) × 1,000,000, using the request rate for your users.')}</p>
      <label class="g-note">Cloud API price per million tokens (edit): $<input class="g-input" id="g-api" type="number" min="0" step="0.1" value="${esc(G.apiPrice)}" style="width:80px"></label>
      <p class="g-note">${cheaper == null ? '' : cheaper >= 0 ? `About ${fmt(cheaper, 0)}% cheaper than that API price.` : `About ${fmt(-cheaper, 0)}% more than that API price.`} The $${esc(G.apiPrice)} default is a placeholder, not a quote.</p></div>
  </div>
  <h3>Every option that fits, cheapest first</h3>
  <div class="g-table-wrap"><table class="g-table"><thead><tr><th></th><th>Hardware</th><th>Devices</th><th>Speed each</th><th>Wait</th><th>$ / month</th><th>$ / M tokens</th></tr></thead><tbody>${opts.map(x => `<tr data-hw="${esc(x.c.hardware_id)}" tabindex="0" class="${x.c.hardware_id === c.hardware_id ? 'sel' : ''}"><td><span class="dot ${x.mark}" title="${x.mark === 'green' ? 'Meets speed and wait targets' : x.mark === 'yellow' ? 'Close: slightly slow' : 'Too slow'}"></span><span class="sr">${x.mark}</span></td><td>${esc(hw(x.c.hardware_id).name)}</td><td>${x.c.device_count}</td><td>${fmt(1 / x.c.tpot_service_s, 0)} tok/s</td><td>${fmt(x.c.ttft_service_s, 2)} s</td><td>${money(x.c.hourly_usd * HOURS_PER_MONTH, 0)}</td><td>${money(x.c.cost_per_million_output, 2)}</td></tr>`).join('')}</tbody></table></div>
  <p class="g-note"><span class="dot green"></span> meets both targets (20+ tokens/s, wait under 2 s) · <span class="dot yellow"></span> close · <span class="dot red"></span> under 10 tokens/s or over 5 s. All values are <span class="tag tag-calculated">calculated</span> from published specs; benchmark to confirm.</p>
  ${missing.length ? `<p class="g-note">${missing.length} catalog entr${missing.length > 1 ? 'ies are' : 'y is'} not shown because specs or prices are “not published”: ${esc(missing.slice(0, 6).map(r => r.name).join(', '))}${missing.length > 6 ? '…' : ''}. Fill them in under Advanced → Workload → Hardware catalog.</p>` : ''}`;
  el.innerHTML = html;
}
function missingForm(r) {
  const need = r.missing.map(k => k === 'price' ? 'a price per hour' : k.replace(/_/g, ' '));
  return `<div class="g-card" style="margin-bottom:16px"><h3>${esc(r.name)} needs ${esc(need.join(' and '))}</h3><p class="g-note">Your catalog lists these as “not published”, so they are left blank rather than guessed. Enter them to include this machine.</p>
  <form id="g-missing-form" data-id="${esc(r.id)}" class="g-model-form">${r.missing.includes('bf16_tflops_dense') ? '<label>BF16 TFLOPS <input class="g-input" name="bf16_tflops_dense" type="number" min="0.1" step="any" required></label>' : ''}${r.missing.includes('memory_gb') ? '<label>Memory GB <input class="g-input" name="memory_gb" type="number" min="1" step="any" required></label>' : ''}${r.missing.includes('memory_bandwidth_gb_s') ? '<label>Bandwidth GB/s <input class="g-input" name="memory_bandwidth_gb_s" type="number" min="1" step="any" required></label>' : ''}${r.missing.includes('price') ? '<label>Price $ per hour <input class="g-input" name="cloud_hourly_usd" type="number" min="0" step="any" required></label>' : ''}<button class="button primary" type="submit">Add to comparison</button></form></div>`;
}

// ---- Stack
function stackLayers() {
  const o = chosen(options()); if (!o) return [];
  const c = o.c, h = hw(c.hardware_id), peak = state.workload.peak_concurrent_users, many = c.replicas > 1;
  const small = peak <= 1 && ['desktop_appliance', 'workstation_gpu'].includes(h.category);
  const platform = h.compute_platform || (/nvidia/i.test(h.name) ? 'CUDA' : h.kind === 'CPU' ? 'CPU kernels (oneDNN / AMX)' : 'Vendor GPU runtime');
  const platKey = {CUDA: 'cuda', ROCm: 'rocm', oneAPI: 'oneapi'}[platform];
  return [
    {id: 'users', name: 'Users', product: `${fmt(G.totalUsers, 0)} people, about ${fmt(peak, 0)} at once`, what: 'The people or apps sending questions.', why: `You said ${fmt(G.totalUsers, 0)} users with ${G.activePct}% active together, so the stack is sized for about ${fmt(peak, 0)} simultaneous requests.`},
    {id: 'gateway', name: 'Gateway', product: many ? 'Inference gateway (Kubernetes Gateway API)' : 'Direct connection (OpenAI-compatible API)', what: 'The front door: it receives requests, checks who is asking, and passes them on.', why: many ? 'With several servers, one front door must share requests among them.' : 'With a single server your apps can call it directly; a simple proxy is optional.'},
    {id: 'scheduler', name: 'Scheduler', product: many ? 'llm-d' : 'None needed', what: 'The host who seats each request at the best server, preferring one that already holds its cached prompt.', why: many ? `${c.replicas} replicas are needed, so something has to decide where each request goes.` : 'One server has nowhere else to send work, so no scheduler is needed.'},
    {id: 'engine', name: 'Serving engine', product: small ? 'llama.cpp or Ollama' : 'vLLM', what: 'The kitchen: loads the model, batches requests, and manages the KV cache.', why: small ? 'A single small box for one person is best served by a simple, light engine.' : 'vLLM is built to serve many people at once on one server by batching their requests.'},
    {id: 'platform', name: 'Compute platform', product: platform, what: 'The vendor software that lets the engine use the chip.', why: `${h.name} is made by ${h.vendor || 'its vendor'}, and its chips are driven through ${platform}.`},
    {id: 'hardware', name: 'Hardware', product: `${c.device_count} × ${h.name}`, what: 'The chips and memory that do the work.', why: `The lowest-cost option that meets your speed and wait targets (${o.mark === 'green' ? 'both met' : 'closest available'}).`},
  ];
}
function renderStack() {
  const el = document.getElementById('g-stack'); if (!el) return;
  const layers = stackLayers();
  if (!layers.length) {el.innerHTML = '<p class="g-sub">Pick a job and a model to see the stack.</p>'; return;}
  const open = layers.find(l => l.id === G.layer) || layers[0];
  el.innerHTML = `<div class="g-stack">${layers.map(l => `<button type="button" class="g-layer" data-g="layer" data-id="${l.id}" aria-pressed="${l.id === open.id}"><b>${l.name}</b><small>${esc(l.product)}</small></button>`).join('')}</div>
  <div class="g-card g-layer-detail" aria-live="polite"><h3>${open.name}: ${esc(open.product)}</h3><p>${esc(open.what)}</p><p><b>Why this one:</b> ${esc(open.why)}</p></div>`;
}

// ---- Static content
function art(kind) {
  const s = (inner) => `<svg viewBox="0 0 240 70" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">${inner}</svg>`;
  const box = (x, y, w, h, f) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" ${f ? `fill="currentColor" fill-opacity=".18"` : ''}/>`;
  switch (kind) {
    case 'prefill': return s(box(10, 20, 100, 30, 1) + '<text x="20" y="40" font-size="12" fill="currentColor" stroke="none">read all at once</text>' + [130, 160, 190].map(x => box(x, 20, 24, 30)).join('') + '<text x="128" y="64" font-size="10" fill="currentColor" stroke="none">one word at a time</text>');
    case 'kv': return s([0, 1, 2, 3, 4].map(i => box(20 + i * 8, 54 - i * 10, 60, 12, 1)).join('') + box(140, 44, 60, 12, 1) + box(140, 28, 60, 12, 1) + '<text x="100" y="40" font-size="11" fill="currentColor" stroke="none">→ grows</text>');
    case 'precision': return s([['BF16', 16, 60], ['FP8', 8, 36], ['INT4', 4, 20]].map(([n, b, h], i) => box(30 + i * 70, 62 - h, 40, h, 1) + `<text x="${34 + i * 70}" y="68" font-size="10" fill="currentColor" stroke="none">${n}=${b / 8}B</text>`).join(''));
    case 'batch': return s([0, 1, 2, 3, 4].map(i => `<circle cx="${20 + i * 20}" cy="35" r="7"/>`).join('') + '<path d="M125 35h30M148 28l8 7-8 7"/>' + box(165, 15, 60, 40, 1));
    case 'engines': return s([['llama.cpp', 28], ['Ollama', 22], ['vLLM', 40], ['+ llm-d', 54]].map(([n, h], i) => box(10 + i * 58, 62 - h, 50, h, 1) + `<text x="${14 + i * 58}" y="${58 - h + 4}" font-size="10" fill="currentColor" stroke="none">${n}</text>`).join(''));
    case 'sched': return s('<circle cx="30" cy="22" r="9"/><path d="M30 31v14M20 38h20"/>' + box(90, 10, 40, 50, 1) + box(150, 10, 40, 50, 1) + '<path d="M44 30h40M144 30h0"/>');
    case 'shard': return s([0, 1, 2, 3].map(i => box(15 + i * 55, 20, 45, 30, 1)).join('') + '<path d="M60 35h10M115 35h10M170 35h10"/>');
    case 'disagg': return s(box(15, 12, 80, 46, 1) + box(145, 12, 80, 46, 1) + '<path d="M100 35h40M132 28l8 7-8 7"/><text x="22" y="40" font-size="11" fill="currentColor" stroke="none">prefill</text><text x="156" y="40" font-size="11" fill="currentColor" stroke="none">decode</text>');
  }
  return '';
}
const EXPLAINERS = [
  ['Prefill and decode', 'prefill', `Before the storyteller (the ${T('gpu', 'GPU')}) says anything, it reads your whole question. That is ${T('prefill', 'prefill')}: it happens all at once and sets the wait before the first word (${T('ttft', 'TTFT')}). Then it answers one word at a time. That is ${T('decode', 'decode')}: each word needs a trip through the whole storybook, so it sets how fast words appear (${T('tps', 'tokens per second')}). Long questions lengthen the wait. Big models slow the speed.`],
  ['KV cache: the sticky notes', 'kv', `While answering, the storyteller keeps sticky notes about everything said so far, so it never rereads from page one. These notes are the ${T('kv', 'KV cache')}. Each conversation gets its own pile, and the pile grows with every word. Ten people with long chats means ten big piles. The notes share memory with the storybook, so you can run out of space even when the model alone fits.`],
  ['Precision: BF16, FP8, INT4', 'precision', `The storybook is written in numbers, and each number takes some bits. ${T('bf16', 'BF16')} uses 16, ${T('fp8', 'FP8')} uses 8, ${T('int4', 'INT4')} uses 4. Divide the number in the name by 8 to get bytes: BF16 is 2 bytes, FP8 is 1, INT4 is half a byte. A 30-billion-parameter model is about 60 GB in BF16, 30 GB in FP8, and 15 GB in INT4. Fewer bits save memory and run faster, at a small cost in accuracy.`],
  ['Batching: many readers, one trip', 'batch', `Each new word needs the whole storybook pulled out of memory. That trip is the slow part, not the thinking. So the storyteller serves many readers on the same trip: one pull of the book, one new word for each of 20 people. That is ${T('batching', 'batching')}. It is why one GPU can serve many people at once, and why each person costs less as more people share it, until the sticky notes fill the memory.`],
  ['Serving engines, small to big', 'engines', `A ${T('engine', 'serving engine')} is the program that runs the model. ${T('llamacpp', 'llama.cpp')} and ${T('ollama', 'Ollama')} are simple and ideal for one person on a laptop or small box. ${T('vllm', 'vLLM')} and ${T('sglang', 'SGLang')} are built for many people on a server GPU; they batch requests and manage KV cache carefully. When one server is not enough, vLLM with ${T('llmd', 'llm-d')} spreads work across many servers on ${T('kubernetes', 'Kubernetes')}.`],
  ['Scheduler versus serving engine', 'sched', `The serving engine is the kitchen: it cooks, meaning it runs the model. The ${T('scheduler', 'scheduler')} is the host: it decides which table each guest sits at, sends returning guests to the table that remembers their order (a cached prompt), and avoids crowded tables. One kitchen needs no host. Many kitchens do. llm-d plays the host for vLLM.`],
  ['Sharding: one model, many GPUs', 'shard', `If the storybook is too big for one storyteller's desk, tear it into parts and give each part to a different GPU. That is ${T('sharding', 'sharding')}. For every word the GPUs must swap notes, many times. Slow links make them wait on each other, so fast links like ${T('nvlink', 'NVLink')} matter. Several GPUs without fast links can run slower than one bigger GPU.`],
  ['Disaggregated and heterogeneous serving', 'disagg', `Reading the question and answering need different strengths: reading needs raw math power, answering needs fast memory. ${T('disaggregated', 'Disaggregated serving')} puts them on separate machines and passes the sticky notes between them. ${T('heterogeneous', 'Heterogeneous serving')} goes further and uses a different chip for each job. It can save money, but adds network and software complexity.`],
];
const VENDORS = [
  ['', 'NVIDIA', 'AMD', 'Intel'],
  ['Hardware', 'H100, H200, B200; L4, L40S; RTX Pro; DGX Spark', 'Instinct MI300X, MI325X, MI355X; Radeon AI Pro', 'Arc Pro B-series (Battlematrix), Gaudi 3, Xeon 6'],
  ['Compute platform', T('cuda', 'CUDA'), T('rocm', 'ROCm'), T('oneapi', 'oneAPI')],
  ['Serving engines', 'vLLM, SGLang, TensorRT-LLM, llama.cpp, Ollama', 'vLLM, SGLang, llama.cpp', 'vLLM, llama.cpp, Ollama'],
  ['Scheduler', 'llm-d, NVIDIA Dynamo', 'llm-d (check current ROCm support)', 'llm-d (check current Intel support)'],
  ['Enterprise packaging', 'NVIDIA AI Enterprise, NIM containers', 'ROCm containers and partner platforms such as Red Hat OpenShift AI', 'Vendor-supplied serving containers and OpenVINO tooling'],
];
function renderStatic() {
  document.getElementById('g-jobs').innerHTML = state.presets.map(p => `<button type="button" class="g-choice" data-g="job" data-id="${p.id}" aria-pressed="${p.id === G.job}"><b>${JOB_ICONS[p.id] || ''} ${esc(p.name)}</b><small>${esc(JOB_LINES[p.id] || p.summary)}</small></button>`).join('');
  document.getElementById('g-quality').innerHTML = Object.entries(PRECISIONS).map(([k, v]) => `<button type="button" class="g-choice" data-g="precision" data-id="${k}" aria-pressed="${k === G.precision}"><b>${v.name} (${T(k, v.code)})</b><small>${v.line}</small></button>`).join('');
  document.getElementById('g-explainers').innerHTML = EXPLAINERS.map(([title, a, text]) => `<div class="g-card"><h3>${title}</h3>${art(a)}<p>${text}</p></div>`).join('') +
    `<div class="g-card wide"><h3>Vendor stacks side by side</h3><div class="g-table-wrap"><table class="g-table" style="white-space:normal"><tbody>${VENDORS.map((row, i) => `<tr>${row.map((cell, j) => i === 0 || j === 0 ? `<th>${cell}</th>` : `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="g-note">Vendor ecosystems change quickly. Check current support before you buy.</p></div>`;
  document.getElementById('g-glossary').innerHTML = Object.entries(GLOSSARY).map(([k, [t, d]]) => `<dt id="g-${k}">${t}</dt><dd>${d}</dd>`).join('');
  syncInputs();
}
function syncInputs() {
  document.getElementById('g-total').value = G.totalUsers; document.getElementById('g-active').value = G.activePct;
  document.getElementById('g-total-out').textContent = fmt(G.totalUsers, 0); document.getElementById('g-active-out').textContent = G.activePct + '%';
  document.getElementById('g-people-out').innerHTML = `About ${lab(fmt(peakUsers(), 0) + ' people', 'calculated', 'Total users × the share active at the same time, rounded, at least 1.')} typing at the same moment`;
  for (const [sel, key] of [['#g-jobs', 'job'], ['#g-quality', 'precision']]) document.querySelectorAll(`${sel} .g-choice`).forEach(b => b.setAttribute('aria-pressed', b.dataset.id === G[key]));
}

// ---- Events
document.addEventListener('click', async (event) => {
  const t = event.target.closest('[data-g]');
  if (t) {
    const a = t.dataset.g;
    if (a === 'job') {G.job = t.dataset.id; syncInputs(); schedule();}
    else if (a === 'precision') {G.precision = t.dataset.id; syncInputs(); schedule();}
    else if (a === 'layer') {G.layer = t.dataset.id; renderStack();}
    else if (a === 'example') await loadExample();
    return;
  }
  const row = event.target.closest('tr[data-hw]');
  if (row) {G.hardwareId = row.dataset.hw; renderResults(); renderStack(); return;}
  const term = event.target.closest('a.term');
  if (term) {
    const dt = document.getElementById('g-' + term.dataset.term);
    document.querySelectorAll('#g-glossary dt.hit').forEach(x => x.classList.remove('hit')); dt?.classList.add('hit');
  }
});
document.addEventListener('keydown', (e) => {if ((e.key === 'Enter' || e.key === ' ') && e.target.matches('tr[data-hw]')) {e.preventDefault(); e.target.click();}});
document.addEventListener('input', (event) => {
  if (event.target.id === 'g-total') {G.totalUsers = Number(event.target.value); syncInputs(); schedule();}
  else if (event.target.id === 'g-active') {G.activePct = Number(event.target.value); syncInputs(); schedule();}
  else if (event.target.id === 'g-api') {G.apiPrice = event.target.value; renderResults(); document.getElementById('g-api')?.focus();}
  else if (event.target.id === 'g-model') {clearTimeout(G.searchTimer); G.searchTimer = setTimeout(() => modelSearch(event.target), 300);}
});
document.addEventListener('focusin', (e) => {if (e.target.id === 'g-model') modelSearch(e.target);});
document.addEventListener('click', (e) => {if (!e.target.closest('#g-model-form')) {const l = document.getElementById('g-model-results'); if (l) l.hidden = true;}});

async function modelSearch(input) {
  const list = document.getElementById('g-model-results');
  try {
    const r = await api('models/search?q=' + encodeURIComponent(input.value.trim()));
    list.innerHTML = r.results.map(m => `<li role="option"><button type="button" class="combo-item" data-pick="${esc(m.id)}">${m.pinned ? '<span class="pin" title="Popular model">★</span>' : ''}<span>${esc(m.id)}</span>${m.downloads != null ? `<small>${fmt(m.downloads, 0)} downloads</small>` : ''}</button></li>`).join('') || '<li class="combo-empty">No matches. You can still enter an ID or link directly.</li>';
  } catch (e) {list.innerHTML = `<li class="combo-empty">${esc(e.message)}</li>`;}
  list.hidden = false; input.setAttribute('aria-expanded', 'true');
}
document.addEventListener('click', async (e) => {
  const pick = e.target.closest('[data-pick]'); if (!pick) return;
  document.getElementById('g-model').value = pick.dataset.pick; document.getElementById('g-model-results').hidden = true;
  await useModel(pick.dataset.pick);
});
async function useModel(id) {
  try {
    state.model = await api('metadata', {model_id: id, revision: 'main'}); state.selectedId = null;
    toast('Model loaded and pinned to its exact version.'); await applyGuided();
  } catch (e) {toast(e.message, true);}
}
document.addEventListener('submit', async (event) => {
  const form = event.target;
  if (form.id === 'g-model-form') {event.preventDefault(); const v = form.elements.q.value.trim(); if (v) await useModel(v);}
  else if (form.id === 'g-missing-form') {
    event.preventDefault();
    try {
      const rec = state.hardware_catalog.map(r => r.id === form.dataset.id ? {...r, ...Object.fromEntries([...new FormData(form)].map(([k, x]) => [k, Number(x)]))} : r);
      const out = await api('catalog/convert', {entries: rec});
      const ids = new Set(rec.map(r => r.id));
      state.hardware = [...state.hardware.filter(h => !ids.has(h.id)), ...out.hardware]; state.hardware_catalog = out.records;
      await applyGuided();
    } catch (e) {toast(e.message, true);}
  }
}, true);
document.getElementById('advanced')?.addEventListener('toggle', (e) => {if (e.target.open && state.workload) render();});

async function loadExample() {
  G.job = 'agents'; G.totalUsers = 40; G.activePct = 10; G.precision = 'fp8'; G.hardwareId = 'nvidia-dgx-spark'; syncInputs();
  await useModel('Qwen/Qwen3-30B-A3B');
  document.getElementById('results').scrollIntoView({behavior: 'smooth'});
}

document.addEventListener('atlas-ready', async () => {
  renderStatic(); renderModelCard();
  await applyGuided();
});
if (typeof state !== 'undefined' && state.presets && state.workload) document.dispatchEvent(new Event('atlas-ready'));
