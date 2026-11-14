# Bug detector frontend

React 18 + Vite + TypeScript + Tailwind UI for the FastAPI backend in `../src/api/app.py`.

## Run it

You need two terminals.

**1. Backend** (Python venv with `requirements.txt` installed). Run it from the project folder
`nlp-code-assistant/`, the one that contains `src/`, not from the git root or from `frontend/`:

```bash
cd nlp-code-assistant
source .venv/bin/activate
uvicorn src.api.app:app --reload --port 8000
```

If you see `ModuleNotFoundError: No module named 'src'`, you're in the wrong folder. Either `cd` into
`nlp-code-assistant/` or add `--app-dir path/to/nlp-code-assistant` to the command.

**2. Frontend** (Node 18.18+):

```bash
cd nlp-code-assistant/frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to `http://localhost:8000/*` and strips the
`/api` prefix, so the backend needs no CORS changes. To point at another backend, set
`BACKEND_URL=http://host:port npm run dev`.

Only models whose artifacts exist can be used. Unavailable models show a gray pill in the top bar and are
disabled in the Analyze tab. Train them as described in the main README.

## Scripts

| Script                 | What it does                                                           |
| ---------------------- | ---------------------------------------------------------------------- |
| `npm run dev`          | Sync metrics, then start the dev server with the `/api` proxy          |
| `npm run build`        | Sync metrics, typecheck, and build to `dist/`                          |
| `npm run preview`      | Serve `dist/` (same `/api` proxy as dev)                               |
| `npm run sync-metrics` | Copy `../results/model_comparison.json` → `src/data/` for the Models tab |
| `npm run typecheck`    | `tsc` with no emit                                                     |
| `npm run lint`         | ESLint                                                                 |
| `npm test`             | Vitest + React Testing Library                                         |

The Models tab reads a build-time snapshot of the comparison file. Run `npm run sync-metrics` (or restart
`dev`) after re-running the model comparison.

## Layout

```
src/
  api.ts            Typed fetch client: omits null optionals, maps errors to short messages
  types.ts          TypeScript mirrors of the Pydantic request/response models
  components/       Tabs, result card, editor, feedback, retrain panel
  lib/              Formatting, examples, health polling, dark-mode hook
  data/             Synced model_comparison.json
  test/             Vitest tests
```

## Notes

- Requests follow the backend's strict Pydantic models (`extra="forbid"`): optional fields are omitted
  rather than sent as `null`, and bodies over 128,000 bytes are rejected before sending.
- `POST /feedback/retrain` runs synchronously and can take minutes. The client waits up to 20 minutes and the
  proxy timeout matches. Keep the tab open while it runs.
- Analyses with a bug probability from 40% to 60% are logged by the backend to
  `data/feedback/uncertain_cases.jsonl`. Feedback goes to `data/feedback/user_feedback.jsonl`. Set
  `BUG_FEEDBACK_DIR` when starting uvicorn to write these somewhere else, for example while testing.
