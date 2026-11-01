# Deploying to Vercel

The same FastAPI app runs locally (`./run.sh`) and on Vercel's Python runtime. Nothing in `planner/` changes
between the two; only where data is stored.

| | Local (`./run.sh`) | Vercel |
|---|---|---|
| Storage mode | `local` (default) | `vercel` (auto-detected from Vercel's `VERCEL` variable, or set `STORAGE_MODE=vercel`) |
| Scenarios, benchmarks, jobs | SQLite at `.local/planner.sqlite3` | The browser (IndexedDB), with **Export my data** / **Import data** JSON buttons |
| Hardware edits | Session only | The browser (IndexedDB) |
| Model metadata cache | SQLite | `/tmp/inference-atlas-cache.sqlite3`, disposable |
| External benchmark adapter | Optional (`BENCHMARK_ADAPTER`) | Disabled |
| Server secret | `HF_TOKEN` (optional) | `HF_TOKEN` (the only one) |

Because the server keeps no state in Vercel mode, data lives in one browser on one device. Use **Export my data**
to back it up or move it.

## What is configured

- `app.py` (repo root): Vercel detects it as the FastAPI entrypoint and serves every non-static path from it,
  with original paths (`/api/...`). There is deliberately no `api/` folder or `/api/*` route in `vercel.json`:
  a rewrite to a separate function changes the path FastAPI sees and every call returns 404.
- `vercel.json`: serves `web/` as static files (`outputDirectory`) and maps `/static/*` to the static root so the
  page's `/static/...` URLs resolve.
- `requirements.txt`: runtime dependencies, exported from `uv.lock`. Regenerate after changing dependencies:
  `uv export --no-dev --no-hashes --no-emit-project | grep -v '^#' > requirements.txt`
- `.vercelignore`: keeps `.venv`, tests, docs and local databases out of the upload.

## Deploy from the command line

Run from this `app/` folder (it contains `vercel.json`).

```sh
npm i -g vercel          # once
vercel login             # once
vercel link              # create or link a project; accept the detected settings
vercel env add HF_TOKEN production    # paste a Hugging Face *read* token when prompted
vercel env add HF_TOKEN preview       # optional, for preview deployments
vercel deploy            # preview deployment
vercel deploy --prod     # production
```

Open the printed URL. `GET /api/health` should return `{"status":"ok",...}`.

## Deploy from the dashboard

1. Push the repository to GitHub, GitLab or Bitbucket.
2. In Vercel choose **Add New → Project** and import it.
3. Set **Root Directory** to `app` (this repository keeps the app in a subfolder). Leave the framework preset as
   **Other** and leave build and install commands empty.
4. Under **Environment Variables** add `HF_TOKEN` with a Hugging Face read token, for Production (and Preview
   if wanted). Mark it **Sensitive**.
5. Click **Deploy**. To change the token later: **Settings → Environment Variables**, edit it, then redeploy
   (variables apply to new deployments only).

## HF_TOKEN

- It is read on the server only. It is never sent to the browser, logged, exported, or included in error
  messages (see `tests/test_api.py`). The page only learns whether a token is configured.
- Gated models such as Llama load through it, but the token's owner must also have been granted access on the
  model's Hugging Face page.
- Create it at https://huggingface.co/settings/tokens with the **Read** scope.

## Verify

- `GET /api/bootstrap` returns `"storage_mode": "vercel"` and `"hf_token_configured": true`.
- Search for a model and inspect it; import a benchmark, reload the page, and confirm it is still listed.
- The external benchmark service section reports that it is disabled.

## Local path is unchanged

`./run.sh` still starts the app on http://127.0.0.1:8765 with SQLite. Without `VERCEL` or `STORAGE_MODE` set,
local mode is used, and the full test suite (`uv run pytest -q`) covers both modes.

## Verified

Deployed with the Vercel CLI (Vercel Python 3.12, dependencies from `uv.lock`): `/`, `/static/*`, `/api/health`,
`/api/bootstrap` (`storage_mode: vercel`), `/api/models/search`, `/api/docs` and `/api/compare` all respond.
Deployment Protection is on by default for new projects; turn it off under **Settings → Deployment Protection**
for a public URL. Browser-side IndexedDB persistence was verified locally in Chrome, not on the deployed site.
