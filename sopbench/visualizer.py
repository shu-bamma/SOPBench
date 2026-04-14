"""Local web server for visualizing step boundary detection results."""

import json
import http.server
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
VIDEOS = ROOT / "videos"

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SOPBench — Step Boundary Visualizer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
         background: #0f1117; color: #e0e0e0; }

  .header { padding: 12px 24px; background: #161b22; border-bottom: 1px solid #30363d;
            display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 16px; font-weight: 600; color: #58a6ff; }
  .header select { background: #21262d; color: #e0e0e0; border: 1px solid #30363d;
                   padding: 6px 12px; border-radius: 6px; font-size: 13px; cursor: pointer; }
  .header .nav-btn { background: #21262d; color: #e0e0e0; border: 1px solid #30363d;
                     padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .header .nav-btn:hover { background: #30363d; }

  .main { display: flex; height: calc(100vh - 49px); }

  .left-panel { flex: 1; padding: 16px; overflow-y: auto; }
  .right-panel { width: 280px; padding: 16px; border-left: 1px solid #30363d;
                 overflow-y: auto; background: #161b22; }

  .video-container { background: #000; border-radius: 8px; overflow: hidden;
                     margin-bottom: 16px; max-height: 400px; display: flex;
                     justify-content: center; }
  video { max-width: 100%; max-height: 400px; }

  .timeline-section { margin-bottom: 12px; }
  .timeline-label { font-size: 11px; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-bottom: 4px; padding-left: 4px; }
  .timeline-label.gt { color: #58a6ff; }
  .timeline-label.pred { color: #f0883e; }

  .timeline-wrapper { position: relative; height: 36px; background: #21262d;
                      border-radius: 6px; overflow: hidden; cursor: pointer; }
  .segment { position: absolute; top: 2px; height: 32px; border-radius: 4px;
             display: flex; align-items: center; padding: 0 6px; overflow: hidden;
             font-size: 10px; font-weight: 500; white-space: nowrap;
             transition: opacity 0.15s; cursor: pointer; }
  .segment:hover { opacity: 0.85; }
  .segment .seg-text { overflow: hidden; text-overflow: ellipsis; }
  .segment.error { border: 2px solid #f85149; }

  .playhead { position: absolute; top: 0; width: 2px; height: 100%;
              background: #fff; pointer-events: none; z-index: 10;
              transition: left 0.05s linear; }

  .time-axis { position: relative; height: 20px; margin-top: 4px; margin-bottom: 16px; }
  .time-tick { position: absolute; top: 0; font-size: 9px; color: #6e7681;
               transform: translateX(-50%); }
  .time-tick::before { content: ''; position: absolute; top: -4px; left: 50%;
                       width: 1px; height: 4px; background: #30363d; }

  .tooltip { position: fixed; background: #2d333b; border: 1px solid #444c56;
             padding: 8px 12px; border-radius: 6px; font-size: 12px; z-index: 100;
             pointer-events: none; display: none; max-width: 350px; }
  .tooltip .tt-desc { font-weight: 600; margin-bottom: 4px; }
  .tooltip .tt-time { color: #8b949e; }
  .tooltip .tt-iou { margin-top: 4px; font-weight: 600; }

  .metric-group { margin-bottom: 16px; }
  .metric-group h3 { font-size: 12px; font-weight: 600; color: #8b949e;
                     text-transform: uppercase; letter-spacing: 0.5px;
                     margin-bottom: 8px; border-bottom: 1px solid #30363d;
                     padding-bottom: 4px; }
  .metric-row { display: flex; justify-content: space-between; padding: 3px 0;
                font-size: 13px; }
  .metric-row .label { color: #8b949e; }
  .metric-row .value { font-weight: 600; font-variant-numeric: tabular-nums; }

  .step-iou-list { list-style: none; }
  .step-iou-item { display: flex; align-items: center; gap: 8px; padding: 4px 0;
                   font-size: 12px; border-bottom: 1px solid #21262d; cursor: pointer; }
  .step-iou-item:hover { background: #21262d; }
  .iou-bar-bg { width: 60px; height: 6px; background: #21262d; border-radius: 3px;
                flex-shrink: 0; }
  .iou-bar { height: 100%; border-radius: 3px; }
  .step-desc { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
               color: #8b949e; }
  .step-iou-val { font-variant-numeric: tabular-nums; width: 36px; text-align: right;
                  flex-shrink: 0; }

  .info-tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
              font-size: 11px; font-weight: 600; margin-bottom: 8px; }
  .info-tag.recipe { background: #1f3a2d; color: #3fb950; }
  .info-tag.error-tag { background: #3d1f20; color: #f85149; }
  .info-tag.correct-tag { background: #1f3a2d; color: #3fb950; }
  .info-tag.model-tag { background: #1f2a3d; color: #58a6ff; }

  .no-data { padding: 40px; text-align: center; color: #6e7681; }
</style>
</head>
<body>

<div class="header">
  <h1>SOPBench</h1>
  <button class="nav-btn" onclick="navVideo(-1)">&#9664; Prev</button>
  <select id="video-select" onchange="loadVideo(this.value)"></select>
  <button class="nav-btn" onclick="navVideo(1)">Next &#9654;</button>
</div>

<div class="main">
  <div class="left-panel">
    <div class="video-container">
      <video id="player" controls></video>
    </div>

    <div class="timeline-section">
      <div class="timeline-label gt">Ground Truth</div>
      <div class="timeline-wrapper" id="gt-timeline" onclick="seekFromTimeline(event, this)">
        <div class="playhead" id="gt-playhead"></div>
      </div>
    </div>

    <div class="timeline-section">
      <div class="timeline-label pred">Gemini Predictions</div>
      <div class="timeline-wrapper" id="pred-timeline" onclick="seekFromTimeline(event, this)">
        <div class="playhead" id="pred-playhead"></div>
      </div>
    </div>

    <div class="time-axis" id="time-axis"></div>
  </div>

  <div class="right-panel" id="metrics-panel">
    <div class="no-data">Select a video to view results</div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const COLORS_GT = [
  '#1f6feb', '#238636', '#8957e5', '#bf8700', '#da3633',
  '#3fb950', '#58a6ff', '#d29922', '#f778ba', '#56d4dd',
  '#7ee787', '#a5d6ff', '#ffa657', '#ff7b72', '#d2a8ff',
];
const COLORS_PRED = [
  '#1a4fa0', '#1a6b2c', '#6e3dba', '#9a6d00', '#b02a28',
  '#2f9940', '#4090e0', '#b08018', '#d060a0', '#44b8c0',
  '#66c870', '#88bae0', '#e09040', '#e06058', '#b890e0',
];

let allResults = [];
let currentData = null;
let currentMaxTime = 1;

async function init() {
  const resp = await fetch('/api/results');
  allResults = await resp.json();
  const sel = document.getElementById('video-select');
  while (sel.firstChild) sel.removeChild(sel.firstChild);
  allResults.forEach(function(r, i) {
    const opt = document.createElement('option');
    opt.value = i;
    const tag = r.dataset === 'captaincook4d' ? '[CC4D]' : '[COIN]';
    opt.textContent = tag + ' ' + r.recording_id + ' \u2014 ' + r.video;
    sel.appendChild(opt);
  });
  if (allResults.length > 0) loadVideo(0);
}

function loadVideo(index) {
  index = parseInt(index);
  document.getElementById('video-select').value = index;
  currentData = allResults[index];
  var d = currentData;

  // Video
  var player = document.getElementById('player');
  player.src = '/video/' + d.dataset + '/' + d.video;
  player.load();

  // Compute max time
  var times = [];
  d.ground_truth.forEach(function(s) { if (s.end_time > 0) times.push(s.end_time); });
  d.predictions.forEach(function(s) { if (s.end_time > 0) times.push(s.end_time); });
  currentMaxTime = (times.length > 0 ? Math.max.apply(null, times) : 1) * 1.05;

  renderTimeline('gt-timeline', d.ground_truth, currentMaxTime, COLORS_GT, false);
  renderTimeline('pred-timeline', d.predictions, currentMaxTime, COLORS_PRED, true);
  renderTimeAxis(currentMaxTime);
  renderMetrics(d);

  // Sync playhead
  player.ontimeupdate = function() {
    var pct = (player.currentTime / currentMaxTime) * 100;
    document.getElementById('gt-playhead').style.left = pct + '%';
    document.getElementById('pred-playhead').style.left = pct + '%';
  };
}

function renderTimeline(containerId, steps, maxTime, colors, isPred) {
  var container = document.getElementById(containerId);
  var playhead = container.querySelector('.playhead');
  // Remove all children except playhead
  while (container.firstChild) container.removeChild(container.firstChild);
  container.appendChild(playhead);

  steps.forEach(function(s, i) {
    if (s.start_time < 0 || s.end_time < 0) return;
    var left = (s.start_time / maxTime) * 100;
    var width = ((s.end_time - s.start_time) / maxTime) * 100;
    var seg = document.createElement('div');
    seg.className = 'segment' + (s.has_errors ? ' error' : '');
    seg.style.left = left + '%';
    seg.style.width = Math.max(width, 0.5) + '%';
    seg.style.background = colors[i % colors.length];

    var span = document.createElement('span');
    span.className = 'seg-text';
    span.textContent = shortDesc(s.description);
    seg.appendChild(span);

    // Tooltip data
    var iouVal = isPred && currentData.metrics.per_step_iou[i] !== undefined
      ? currentData.metrics.per_step_iou[i] : null;

    (function(step, iou) {
      seg.addEventListener('mouseenter', function(e) { showTooltip(e, step, iou); });
      seg.addEventListener('mousemove', moveTooltip);
      seg.addEventListener('mouseleave', hideTooltip);
      seg.addEventListener('click', function(e) {
        e.stopPropagation();
        document.getElementById('player').currentTime = step.start_time;
      });
    })(s, iouVal);

    container.appendChild(seg);
  });
}

function renderTimeAxis(maxTime) {
  var axis = document.getElementById('time-axis');
  while (axis.firstChild) axis.removeChild(axis.firstChild);
  var step = maxTime < 60 ? 10 : maxTime < 180 ? 15 : maxTime < 600 ? 30 : 60;
  for (var t = 0; t <= maxTime; t += step) {
    var tick = document.createElement('div');
    tick.className = 'time-tick';
    tick.style.left = (t / maxTime * 100) + '%';
    tick.textContent = formatTime(t);
    axis.appendChild(tick);
  }
}

function renderMetrics(d) {
  var panel = document.getElementById('metrics-panel');
  // Clear panel
  while (panel.firstChild) panel.removeChild(panel.firstChild);

  var m = d.metrics;

  // Info tags
  var tagsDiv = document.createElement('div');
  tagsDiv.style.marginBottom = '12px';

  if (d.dataset === 'captaincook4d') {
    var recTag = document.createElement('span');
    recTag.className = 'info-tag recipe';
    recTag.textContent = d.recording_id;
    tagsDiv.appendChild(recTag);
    tagsDiv.appendChild(document.createTextNode(' '));

    var isErr = d.ground_truth.some(function(s) { return s.has_errors; });
    var errTag = document.createElement('span');
    errTag.className = isErr ? 'info-tag error-tag' : 'info-tag correct-tag';
    errTag.textContent = isErr ? 'Has Errors' : 'Correct';
    tagsDiv.appendChild(errTag);
    tagsDiv.appendChild(document.createTextNode(' '));
  }

  var modelTag = document.createElement('span');
  modelTag.className = 'info-tag model-tag';
  modelTag.textContent = d.model;
  tagsDiv.appendChild(modelTag);
  panel.appendChild(tagsDiv);

  // Aggregate metrics
  var aggGroup = document.createElement('div');
  aggGroup.className = 'metric-group';
  var aggTitle = document.createElement('h3');
  aggTitle.textContent = 'Aggregate Metrics';
  aggGroup.appendChild(aggTitle);

  var metricDefs = [
    ['Mean IoU', m.mean_iou, iouColor(m.mean_iou)],
    ['R@1 (IoU\u22650.3)', m['recall_at_1_iou_0.3'], null],
    ['R@1 (IoU\u22650.5)', m['recall_at_1_iou_0.5'], null],
    ['R@1 (IoU\u22650.7)', m['recall_at_1_iou_0.7'], null],
    ['Detection Rate', m.step_detection_rate, null],
    ['Ordering', m.ordering_compliance, null],
  ];

  metricDefs.forEach(function(def) {
    var row = document.createElement('div');
    row.className = 'metric-row';
    var lbl = document.createElement('span');
    lbl.className = 'label';
    lbl.textContent = def[0];
    var val = document.createElement('span');
    val.className = 'value';
    val.textContent = (def[1] * 100).toFixed(1) + '%';
    if (def[2]) val.style.color = def[2];
    row.appendChild(lbl);
    row.appendChild(val);
    aggGroup.appendChild(row);
  });

  // Steps count
  var stepsRow = document.createElement('div');
  stepsRow.className = 'metric-row';
  var stepsLbl = document.createElement('span');
  stepsLbl.className = 'label';
  stepsLbl.textContent = 'Steps (GT / Pred)';
  var stepsVal = document.createElement('span');
  stepsVal.className = 'value';
  stepsVal.textContent = m.num_gt_steps + ' / ' + m.num_predictions;
  stepsRow.appendChild(stepsLbl);
  stepsRow.appendChild(stepsVal);
  aggGroup.appendChild(stepsRow);
  panel.appendChild(aggGroup);

  // Per-step IoU
  var stepGroup = document.createElement('div');
  stepGroup.className = 'metric-group';
  var stepTitle = document.createElement('h3');
  stepTitle.textContent = 'Per-Step IoU';
  stepGroup.appendChild(stepTitle);

  var stepList = document.createElement('ul');
  stepList.className = 'step-iou-list';

  d.ground_truth.forEach(function(gt, i) {
    if (gt.start_time < 0) return;
    var iou = m.per_step_iou[i] !== undefined ? m.per_step_iou[i] : 0;
    var color = iouColor(iou);

    var li = document.createElement('li');
    li.className = 'step-iou-item';
    li.addEventListener('click', function() {
      document.getElementById('player').currentTime = gt.start_time;
    });

    var barBg = document.createElement('div');
    barBg.className = 'iou-bar-bg';
    var bar = document.createElement('div');
    bar.className = 'iou-bar';
    bar.style.width = (iou * 100) + '%';
    bar.style.background = color;
    barBg.appendChild(bar);
    li.appendChild(barBg);

    var desc = document.createElement('span');
    desc.className = 'step-desc';
    desc.title = gt.description;
    desc.textContent = shortDesc(gt.description);
    if (gt.has_errors) {
      var warn = document.createElement('span');
      warn.style.color = '#f85149';
      warn.textContent = ' \u26A0';
      desc.appendChild(warn);
    }
    li.appendChild(desc);

    var valSpan = document.createElement('span');
    valSpan.className = 'step-iou-val';
    valSpan.style.color = color;
    valSpan.textContent = (iou * 100).toFixed(0) + '%';
    li.appendChild(valSpan);

    stepList.appendChild(li);
  });

  stepGroup.appendChild(stepList);
  panel.appendChild(stepGroup);
}

function seekFromTimeline(e, container) {
  var rect = container.getBoundingClientRect();
  var pct = (e.clientX - rect.left) / rect.width;
  document.getElementById('player').currentTime = pct * currentMaxTime;
}

function navVideo(delta) {
  var sel = document.getElementById('video-select');
  var newIdx = Math.max(0, Math.min(allResults.length - 1, parseInt(sel.value) + delta));
  loadVideo(newIdx);
}

function showTooltip(e, step, iou) {
  var tt = document.getElementById('tooltip');
  while (tt.firstChild) tt.removeChild(tt.firstChild);

  var descEl = document.createElement('div');
  descEl.className = 'tt-desc';
  descEl.textContent = step.description;
  tt.appendChild(descEl);

  var timeEl = document.createElement('div');
  timeEl.className = 'tt-time';
  timeEl.textContent = formatTime(step.start_time) + ' \u2192 ' + formatTime(step.end_time);
  tt.appendChild(timeEl);

  if (iou !== null) {
    var iouEl = document.createElement('div');
    iouEl.className = 'tt-iou';
    iouEl.style.color = iouColor(iou);
    iouEl.textContent = 'IoU: ' + (iou * 100).toFixed(1) + '%';
    tt.appendChild(iouEl);
  }

  if (step.confidence !== undefined) {
    var confEl = document.createElement('div');
    confEl.className = 'tt-time';
    confEl.textContent = 'Confidence: ' + (step.confidence * 100).toFixed(0) + '%';
    tt.appendChild(confEl);
  }

  tt.style.display = 'block';
  moveTooltip(e);
}

function moveTooltip(e) {
  var tt = document.getElementById('tooltip');
  tt.style.left = (e.clientX + 12) + 'px';
  tt.style.top = (e.clientY - 10) + 'px';
}

function hideTooltip() {
  document.getElementById('tooltip').style.display = 'none';
}

function shortDesc(desc) {
  var m = desc.match(/^[A-Za-z]+-(.+)/);
  var clean = m ? m[1] : desc;
  return clean.length > 30 ? clean.slice(0, 28) + '...' : clean;
}

function formatTime(s) {
  if (s < 0) return '--:--';
  var m = Math.floor(s / 60);
  var sec = Math.floor(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}

function iouColor(v) {
  if (v >= 0.7) return '#3fb950';
  if (v >= 0.5) return '#d29922';
  if (v >= 0.3) return '#f0883e';
  return '#f85149';
}

init();
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            self._serve_html()
        elif path == "/api/results":
            self._serve_results()
        elif path.startswith("/video/"):
            self._serve_video(path)
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def _serve_results(self):
        """Collect all result JSONs from results/ directory."""
        results = []
        if RESULTS.exists():
            for dataset_dir in sorted(RESULTS.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                for model_dir in sorted(dataset_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    for json_file in sorted(model_dir.glob("*.json")):
                        if json_file.name.startswith("_"):
                            continue
                        with open(json_file) as f:
                            results.append(json.load(f))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(results).encode())

    def _serve_video(self, path):
        """Serve video files. Path format: /video/{dataset}/{filename}"""
        parts = path.split("/")[2:]
        if len(parts) < 2:
            self.send_error(404)
            return

        dataset = parts[0]
        filename = "/".join(parts[1:])
        dir_map = {
            "captaincook4d": VIDEOS / "captaincook4d_samples",
            "coin": VIDEOS / "coin_samples",
        }
        video_dir = dir_map.get(dataset)
        if not video_dir:
            self.send_error(404)
            return

        video_path = video_dir / filename
        if not video_path.exists() or not video_path.is_file():
            self.send_error(404, "Video not found: " + filename)
            return

        file_size = video_path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            byte_range = range_header.strip().split("=")[1]
            start_str, end_str = byte_range.split("-")
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range",
                             "bytes %d-%d/%d" % (start, end, file_size))
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(video_path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(video_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def log_message(self, format, *args):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SOPBench results visualizer")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = http.server.HTTPServer(("", args.port), Handler)
    print("SOPBench Visualizer running at http://localhost:%d" % args.port)
    print("Results dir: %s" % RESULTS)
    print("Videos dir: %s" % VIDEOS)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
