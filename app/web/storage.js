'use strict';
// Browser storage for Vercel mode (stateless server): scenarios, benchmark records and hardware edits
// live in IndexedDB, with JSON export/import. In local mode none of this runs; SQLite stays in charge.
const Atlas = (() => {
  const DB = 'inference-atlas', STORES = ['scenarios', 'benchmarks', 'hardware'];
  let dbp;
  function open() {
    dbp ||= new Promise((resolve, reject) => {
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => {for (const s of STORES) if (!req.result.objectStoreNames.contains(s)) req.result.createObjectStore(s);};
      req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error);
    });
    return dbp;
  }
  async function tx(store, mode, fn) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, mode), out = fn(t.objectStore(store));
      t.oncomplete = () => resolve(out.result); t.onerror = () => reject(t.error);
    });
  }
  const put = (store, key, value) => tx(store, 'readwrite', s => s.put(value, key));
  const get = (store, key) => tx(store, 'readonly', s => s.get(key));
  const clear = (store) => tx(store, 'readwrite', s => s.clear());
  async function entries(store) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const out = {}, req = db.transaction(store).objectStore(store).openCursor();
      req.onsuccess = () => {const c = req.result; if (c) {out[c.key] = c.value; c.continue();} else resolve(out);};
      req.onerror = () => reject(req.error);
    });
  }
  async function exportAll() {
    return {format: 'inference-atlas-browser-data', version: 1, exported_at: new Date().toISOString(),
      scenarios: await entries('scenarios'), benchmarks: Object.values(await entries('benchmarks')), hardware: (await get('hardware', 'catalog')) || null};
  }
  async function importAll(data) {
    if (!data || data.format !== 'inference-atlas-browser-data' || data.version !== 1) throw new Error('Not an Inference Atlas export file.');
    for (const [name, value] of Object.entries(data.scenarios || {})) await put('scenarios', name, value);
    for (const run of data.benchmarks || []) await put('benchmarks', run.run_id, run);
    if (data.hardware) await put('hardware', 'catalog', data.hardware);
  }
  return {put, get, clear, entries, exportAll, importAll};
})();
