"""
python image_compare_web.py \
  --root /mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref/ \
  --output-dir /mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_sref/compare \
  --output-name selections.jsonl \
  --images-per-row 2 \
  --only-unlabeled \
  --host 0.0.0.0 \
  --port 7860 \
  --meta-json /mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_sref/instruction.json

python image_compare_web.py \
  --root /data/benchmark_metrics/sample_1500_bench_cref_sref \
  --output-dir /data/benchmark_metrics/sample_1500_bench_cref_sref/compare \
  --output-name selections.jsonl \
  --images-per-row 2 \
  --only-unlabeled \
  --host 0.0.0.0 \
  --port 7860 \
  --meta-json /data/benchmark_metrics/sample_1500_bench_cref_sref/prompts.json
"""
import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Image Compare</title>
  <style>
    :root { --fg:#111; --muted:#666; --bg:#fff; --border:#ddd; --sel:#2e7d32; }
    body { margin: 0; font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color: var(--fg); background: var(--bg); }
    header { position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--border); padding: 10px 12px; z-index: 10; }
    .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .grow { flex: 1; min-width: 220px; }
    .basename { font-size: 16px; font-weight: 600; user-select: text; }
    .meta { margin-top: 6px; color: var(--muted); font-size: 13px; white-space: pre-wrap; user-select: text; }
    .controls button { margin-right: 8px; }
    button { cursor: pointer; border: 1px solid var(--border); background: #f7f7f7; padding: 6px 10px; border-radius: 6px; }
    button:hover { background: #efefef; }
    .progress { height: 10px; background: #e0e0e0; border-radius: 999px; overflow: hidden; margin-top: 10px; }
    .progress > div { height: 100%; width: 0%; background: var(--sel); }
    main { padding: 12px; }
    .grid { display: grid; gap: 10px; grid-template-columns: repeat(var(--cols), minmax(180px, 1fr)); }
    .card { position: relative; border: 1px solid var(--border); border-radius: 10px; padding: 8px; background: #fff; }
    .card.selected { border-color: var(--sel); box-shadow: 0 0 0 2px rgba(46,125,50,0.25); }
    .card img { width: 100%; height: auto; display: block; border-radius: 8px; background: #000; }
    .label { margin-top: 6px; font-size: 12px; color: var(--muted); user-select: text; white-space: pre-wrap; }
    .zoom { position: absolute; top: 6px; right: 6px; border-radius: 8px; }
    .modal { position: fixed; inset: 0; background: rgba(0,0,0,0.9); display: none; align-items: center; justify-content: center; z-index: 999; }
    .modal.open { display: flex; }
    .modal img { max-width: 96vw; max-height: 92vh; }
    .modal .bar { position: fixed; top: 10px; right: 10px; display: flex; gap: 10px; align-items: center; }
    .modal .bar a, .modal .bar button { color: #fff; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22); }
  </style>
</head>
<body>
  <header>
    <div class="row">
      <div class="grow">
        <div class="basename" id="basename"></div>
        <div class="meta" id="meta"></div>
      </div>
      <div class="controls">
        <button id="prevBtn">上一组</button>
        <button id="skipBtn">跳过</button>
        <button id="saveBtn">记录并下一张</button>
      </div>
      <div class="muted" id="counter" style="color:var(--muted); user-select:text;"></div>
    </div>
    <div class="progress"><div id="bar"></div></div>
  </header>
  <main>
    <div class="grid" id="grid"></div>
  </main>

  <div class="modal" id="modal">
    <div class="bar">
      <a id="openOrig" href="#" target="_blank" rel="noreferrer">打开原图</a>
      <button id="closeModal">关闭</button>
    </div>
    <img id="modalImg" alt="" />
  </div>

  <script>
    const state = { cols: 4, selected: new Set(), currentBasename: null };

    function $(id) { return document.getElementById(id); }

    async function api(path, options) {
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data && data.error ? data.error : ('HTTP ' + res.status));
      return data;
    }

    function render(payload) {
      state.cols = payload.images_per_row || 4;
      document.documentElement.style.setProperty('--cols', String(state.cols));
      state.currentBasename = payload.basename;
      state.selected = new Set(payload.selected || []);

      $('basename').textContent = payload.basename || '';
      const metaLines = (payload.meta_values || []).map(String);
      $('meta').textContent = metaLines.join('\\n');

      $('counter').textContent = (payload.index + 1) + '/' + payload.total + '  剩余:' + (payload.total - payload.index);
      const pct = payload.total > 0 ? Math.floor((payload.index / payload.total) * 100) : 0;
      $('bar').style.width = pct + '%';

      const grid = $('grid');
      grid.innerHTML = '';
      (payload.items || []).forEach(item => {
        const card = document.createElement('div');
        card.className = 'card' + (state.selected.has(item.rel_path) ? ' selected' : '');
        card.dataset.path = item.rel_path;

        const btn = document.createElement('button');
        btn.className = 'zoom';
        btn.textContent = '🔍';
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          openModal(item.img_url);
        });

        const img = document.createElement('img');
        img.src = item.img_url;
        img.loading = 'lazy';
        img.alt = '';

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = item.model_id + '\\n' + item.file_name;

        card.addEventListener('click', () => {
          const p = card.dataset.path;
          if (state.selected.has(p)) state.selected.delete(p);
          else state.selected.add(p);
          card.classList.toggle('selected');
        });

        card.appendChild(btn);
        card.appendChild(img);
        card.appendChild(label);
        grid.appendChild(card);
      });
    }

    async function refresh() {
      const payload = await api('/api/current');
      render(payload);
    }

    async function nav(delta) {
      const payload = await api('/api/nav', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ delta }) });
      render(payload);
    }

    async function save() {
      const selected = Array.from(state.selected);
      if (selected.length === 0) {
        await nav(1);
        return;
      }
      try {
        const payload = await api('/api/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ basename: state.currentBasename, selected, overwrite: false }) });
        render(payload);
      } catch (e) {
        const msg = String(e && e.message ? e.message : e);
        if (msg.includes('conflict')) {
          const ok = confirm('已存在，是否覆盖？');
          if (!ok) return;
          const payload = await api('/api/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ basename: state.currentBasename, selected, overwrite: true }) });
          render(payload);
          return;
        }
        alert(msg);
      }
    }

    function openModal(imgUrl) {
      $('modal').classList.add('open');
      $('modalImg').src = imgUrl;
      $('openOrig').href = imgUrl;
    }

    function closeModal() {
      $('modal').classList.remove('open');
      $('modalImg').src = '';
    }

    $('prevBtn').addEventListener('click', () => nav(-1));
    $('skipBtn').addEventListener('click', () => nav(1));
    $('saveBtn').addEventListener('click', () => save());
    $('closeModal').addEventListener('click', () => closeModal());
    $('modal').addEventListener('click', (e) => { if (e.target === $('modal')) closeModal(); });
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

    refresh();
  </script>
</body>
</html>
"""


def load_meta_map(meta_json_path):
    if meta_json_path is None:
        return {}
    p = Path(meta_json_path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def load_done_keys(output_path):
    if not output_path.exists():
        return set()
    keys = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if isinstance(data, dict):
            keys.update(data.keys())
    return keys


def upsert_jsonl(output_path, key, record, overwrite):
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True, False
    lines = output_path.read_text(encoding="utf-8").splitlines()
    found_index = None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if key in data:
            found_index = i
            break
    if found_index is not None and not overwrite:
        return False, True
    if found_index is not None and overwrite:
        lines[found_index] = json.dumps(record, ensure_ascii=False)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True, False
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True, False


class CompareModel:
    def __init__(self, root_dir, output_dir, output_name, extensions, images_per_row, only_unlabeled, meta_map):
        self.root_dir = Path(root_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_name = output_name
        self.extensions = extensions
        self.images_per_row = images_per_row
        self.only_unlabeled = only_unlabeled
        self.meta_map = meta_map or {}

        self.model_ids = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        if not self.model_ids:
            raise ValueError("未找到任何子文件夹")

        self.per_model = self._build_per_model()
        self.basenames = self._compute_basenames()
        self.index = 0

        if self.only_unlabeled:
            done = load_done_keys(self.output_path)
            if done:
                self.basenames = [b for b in self.basenames if b not in done]

        if not self.basenames:
            raise ValueError("没有可展示的同名图片")

    @property
    def output_path(self):
        return self.output_dir / self.output_name

    def _build_per_model(self):
        per_model = {}
        for model_id in self.model_ids:
            model_dir = self.root_dir / model_id
            stem_map = {}
            for f in model_dir.iterdir():
                if f.is_file() and f.suffix.lower() in self.extensions:
                    stem_map.setdefault(f.stem, []).append(f)
            for stem in stem_map:
                stem_map[stem] = sorted(stem_map[stem], key=lambda p: p.name)
            per_model[model_id] = stem_map
        return per_model

    def _compute_basenames(self):
        required_ids = [m for m in self.model_ids if not m.startswith("one_lora_")]
        if not required_ids:
            required_ids = list(self.model_ids)
        common = None
        for model_id in required_ids:
            stems = set(self.per_model.get(model_id, {}).keys())
            common = stems if common is None else common & stems
        return sorted(common or [])

    def clamp_index(self):
        if self.index < 0:
            self.index = 0
        if self.index >= len(self.basenames):
            self.index = max(len(self.basenames) - 1, 0)

    def nav(self, delta):
        self.index += int(delta)
        if self.index < 0:
            self.index = 0
        if self.index >= len(self.basenames):
            self.index = len(self.basenames) - 1
        return self.current_payload()

    def current_basename(self):
        self.clamp_index()
        return self.basenames[self.index]

    def meta_values(self, basename):
        v = self.meta_map.get(basename)
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    def items_for(self, basename):
        items = []
        for model_id in self.model_ids:
            for path in self.per_model.get(model_id, {}).get(basename, []):
                rel_path = str(path.resolve().relative_to(self.root_dir))
                items.append(
                    {
                        "model_id": model_id,
                        "file_name": path.name,
                        "rel_path": rel_path,
                        "img_url": "/img?path=" + quote(rel_path),
                    }
                )
        return items

    def current_payload(self):
        basename = self.current_basename()
        return {
            "basename": basename,
            "index": self.index,
            "total": len(self.basenames),
            "images_per_row": self.images_per_row,
            "items": self.items_for(basename),
            "selected": [],
            "meta_values": self.meta_values(basename),
        }

    def save_current(self, basename, selected_rel_paths, overwrite):
        current = self.current_basename()
        if basename != current:
            raise ValueError("basename 不匹配")
        abs_paths = []
        for rel in selected_rel_paths:
            p = (self.root_dir / rel).resolve()
            if not str(p).startswith(str(self.root_dir) + os.sep):
                raise ValueError("非法路径")
            abs_paths.append(str(p))
        ok, conflict = upsert_jsonl(self.output_path, basename, {basename: abs_paths}, overwrite=overwrite)
        if conflict:
            return None, True
        if ok:
            if self.index + 1 < len(self.basenames):
                self.index += 1
            return self.current_payload(), False
        raise ValueError("写入失败")


class Handler(BaseHTTPRequestHandler):
    model = None

    def _json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _text(self, status, text, content_type="text/html; charset=utf-8"):
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._text(HTTPStatus.OK, HTML_PAGE)
        if parsed.path == "/api/current":
            try:
                payload = self.model.current_payload()
                return self._json(HTTPStatus.OK, payload)
            except Exception as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        if parsed.path == "/img":
            qs = parse_qs(parsed.query)
            rel = qs.get("path", [""])[0]
            rel = unquote(rel)
            try:
                p = (self.model.root_dir / rel).resolve()
                if not str(p).startswith(str(self.model.root_dir) + os.sep):
                    return self._json(HTTPStatus.BAD_REQUEST, {"error": "非法路径"})
                if not p.exists() or not p.is_file():
                    return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                content_type, _ = mimetypes.guess_type(str(p))
                content_type = content_type or "application/octet-stream"
                data = p.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/nav":
            body = self._read_json()
            try:
                delta = int(body.get("delta", 0))
                payload = self.model.nav(delta)
                return self._json(HTTPStatus.OK, payload)
            except Exception as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        if parsed.path == "/api/save":
            body = self._read_json()
            try:
                basename = body.get("basename")
                selected = body.get("selected") or []
                overwrite = bool(body.get("overwrite"))
                if not isinstance(selected, list):
                    selected = []
                payload, conflict = self.model.save_current(basename, selected, overwrite=overwrite)
                if conflict:
                    return self._json(HTTPStatus.CONFLICT, {"error": "conflict"})
                return self._json(HTTPStatus.OK, payload)
            except Exception as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, fmt, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="selections.jsonl")
    parser.add_argument("--ext", default="png,jpg,jpeg")
    parser.add_argument("--images-per-row", type=int, default=6)
    parser.add_argument("--only-unlabeled", action="store_true")
    parser.add_argument("--meta-json", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    exts = {f".{e.strip().lower()}" for e in args.ext.split(",") if e.strip()}
    meta_map = load_meta_map(args.meta_json)
    model = CompareModel(
        root_dir=args.root,
        output_dir=args.output_dir,
        output_name=args.output_name,
        extensions=exts,
        images_per_row=args.images_per_row,
        only_unlabeled=args.only_unlabeled,
        meta_map=meta_map,
    )
    Handler.model = model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    host_for_print = args.host
    if host_for_print in ("0.0.0.0", "::", ""):
        host_for_print = "127.0.0.1"
    url = f"http://{host_for_print}:{args.port}/"
    print(f"Server listening on {args.host}:{args.port}")
    print(f"Open in browser: {url}")
    print("If using VSCode Remote SSH, forward the port then open the forwarded URL.")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
