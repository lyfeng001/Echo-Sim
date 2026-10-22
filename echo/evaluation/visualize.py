"""Uniformly sample trials and view two camera videos with whole-H50 signals."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil

import numpy as np

from echo.io import atomic_json, fingerprint, read_json, sha256
from echo.evaluation.results import inspect_results
from echo.evaluation.signals import load_metric_config, whole_chunk


def select_trials(run_dir, manifest, seed, count, output):
    rows, status = inspect_results(run_dir, manifest)
    selection_path = output/"selection.json"
    binding = dict(manifest_sha256=sha256(manifest), sample_seed=seed, requested_samples=count)
    if selection_path.exists():
        selection = read_json(selection_path)
        if any(selection[k] != v for k, v in binding.items()):
            raise ValueError("Existing visualization has a different manifest or sample configuration")
    else:
        candidates = sorted(rows)
        if type(count) is not int or not 0 < count <= len(candidates) or type(seed) is not int or seed < 0:
            raise ValueError("Need a positive sample count no larger than completed trials, and a nonnegative seed")
        indices = np.random.default_rng(seed).choice(len(candidates), size=count, replace=False)
        selected = [candidates[int(i)] for i in sorted(indices)]
        selection = dict(**binding, candidate_ids=candidates, selected_ids=selected,
                         sampling="uniform without replacement; no outcome or signal filter",
                         complete_pool=status["complete"], sources={})
        for episode_id in selected:
            directory = Path(run_dir)/"episodes"/episode_id
            names = ("result.json", "trajectory.npz", "signals.npz", "agentview.mp4", "wrist.mp4")
            selection["sources"][episode_id] = {name: sha256(directory/name) for name in names}
        atomic_json(selection_path, selection)
    for episode_id, records in selection["sources"].items():
        if episode_id not in rows:
            raise ValueError("Selected trial is no longer valid")
        for name, expected in records.items():
            if sha256(Path(run_dir)/"episodes"/episode_id/name) != expected:
                raise ValueError("A frozen visualization source has changed")
    return selection, rows


def plot_signals(directory, target, row, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    queries, ss, fs = whole_chunk(directory/"signals.npz", row["steps"])
    with np.load(directory/"trajectory.npz", allow_pickle=False) as payload:
        events = payload["success_after_action"].copy()
    time = queries/config.control_hz
    duration = row["steps"]/config.control_hz
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained", dpi=160)
    ax.step(np.r_[time, duration], np.r_[ss, ss[-1]], where="post", label="SS (whole-H50 mean)", color="#1676b8")
    ax.step(np.r_[time, duration], np.r_[fs, fs[-1]], where="post", label="FS (whole-H50 mean)", color="#d17919")
    ax.axhline(config.ss_threshold, color="#1676b8", linestyle=":", linewidth=1)
    ax.axhline(config.fs_threshold, color="#d17919", linestyle=":", linewidth=1)
    first = row["first_success_step"]
    if first is not None:
        first_time = first/config.control_hz
        ax.axvline(first_time, color="#368047", linestyle="--", label="First environment success")
        ax.axvspan(first_time, min(duration, first_time+config.success_window_seconds), color="#368047", alpha=.1)
    ax.set(xlim=(0, duration), xlabel="Time (s)", ylabel="Signal", title=f'{row["suite"]} | {row["id"]} | '+
           ("Success" if row["success"] else "Timeout"))
    ax.grid(alpha=.15)
    ax.legend(loc="upper right", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(target/("signals."+extension), dpi=160)
    fig.canvas.draw()
    box = ax.get_position()
    bounds = dict(left=box.x0, right=box.x1, top=1-box.y1, bottom=1-box.y0)
    plt.close(fig)
    return dict(id=row["id"], steps=row["steps"], control_hz=config.control_hz,
                query_steps=queries.tolist(), ss=ss.tolist(), fs=fs.tolist(),
                first_success_step=first, success_after_action=events.astype(int).tolist(), plot_bounds=bounds)


def make_gallery(output, selected, records):
    cards = []
    for episode_id in selected:
        row = records[episode_id]
        cards.append(f'''<article data-id="{html.escape(episode_id)}">
<h2>{html.escape(row['suite'])} · {html.escape(episode_id)}</h2>
<p>{html.escape(row.get('instruction', ''))}</p>
<div class="videos"><video class="agent" controls preload="metadata" src="{episode_id}/agentview.mp4"></video>
<video class="wrist" muted preload="metadata" src="{episode_id}/wrist.mp4"></video></div>
<div class="plot"><img src="{episode_id}/signals.png" alt="Whole-H50 mean SS and FS"><span class="cursor"></span></div>
<p class="status"></p><a href="{episode_id}/signals.pdf">PDF</a>
</article>''')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Echo signal visualization</title><style>
body{max-width:1050px;margin:28px auto;padding:0 18px;font:16px system-ui;color:#233142;background:#fafafa}
article{background:white;border:1px solid #ddd;border-radius:8px;padding:18px;margin:24px 0}h2{font-size:18px}
.videos{display:flex;gap:12px}.videos video{width:calc(50% - 6px)}.plot{position:relative}.plot img{width:100%}
.cursor{position:absolute;top:12%;bottom:15%;width:2px;background:#be334c;left:7%;pointer-events:none}
.status{font-variant-numeric:tabular-nums}a{color:#1676b8}
</style><h1>Echo: video and whole-chunk SS / FS</h1>
<p>Uniform random trials, frozen before rendering. Each signal is the mean of all 50 predictions;
values are held until the next policy query. Environment success stays latched after its first occurrence.</p>
''' + "\n".join(cards) + '''<script>
for (const card of document.querySelectorAll('article')) {
  const a=card.querySelector('.agent'), w=card.querySelector('.wrist');
  fetch(card.dataset.id+'/trace.json').then(r=>r.json()).then(trace=>{
    const update=()=>{
      const physical=a.currentTime * trace.video_fps / trace.control_hz;
      const step=Math.min(trace.steps, Math.floor(a.currentTime*trace.video_fps));
      let q=0;while(q+1<trace.query_steps.length && trace.query_steps[q+1]<=step)q++;
      const observed=step>0 ? Boolean(trace.success_after_action[step-1]) : false;
      const latched=trace.first_success_step!==null && step>=trace.first_success_step;
      card.querySelector('.status').textContent=`${physical.toFixed(2)} s | action count ${step} | SS ${trace.ss[q].toFixed(3)} | FS ${trace.fs[q].toFixed(3)} | environment now ${observed} | latched ${latched}`;
      const b=trace.plot_bounds, cursor=card.querySelector('.cursor');
      cursor.style.left=(100*(b.left+(b.right-b.left)*step/trace.steps))+'%';
      cursor.style.top=100*b.top+'%';cursor.style.bottom=100*(1-b.bottom)+'%';
      if(Math.abs(w.currentTime-a.currentTime)>.08)w.currentTime=a.currentTime;
    };
    a.addEventListener('play',()=>{w.currentTime=a.currentTime;w.play().catch(()=>{});});
    a.addEventListener('pause',()=>w.pause());a.addEventListener('seeking',update);a.addEventListener('timeupdate',update);
    const animate=()=>{if(!a.paused)update();requestAnimationFrame(animate);};animate();update();
  });
}
</script></html>'''
    (output/"index.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", default="manifests/evaluation_seed7.json")
    parser.add_argument("--config", default="configs/signals.json")
    parser.add_argument("--output", help="Default: RUN/visualizations")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.run)/"visualizations"
    output.mkdir(parents=True, exist_ok=True)
    settings = read_json(args.config)
    config = load_metric_config(args.config)
    identity_path = Path(args.run)/"identity.json"
    identity = read_json(identity_path) if identity_path.exists() else None
    if identity and identity["protocol"]["control_hz"] != config.control_hz:
        raise ValueError("Signal time scale differs from the evaluation control frequency")
    selection, rows = select_trials(args.run, args.manifest, settings["visualization_sample_seed"],
                                   settings["visualization_samples"], output)
    for episode_id in selection["selected_ids"]:
        directory = Path(args.run)/"episodes"/episode_id
        target = output/episode_id
        target.mkdir(exist_ok=True)
        trace = plot_signals(directory, target, rows[episode_id], config)
        trace["video_fps"] = identity["video_fps"] if identity else config.control_hz
        atomic_json(target/"trace.json", trace)
        for name in ("agentview.mp4", "wrist.mp4"):
            shutil.copyfile(directory/name, target/name)
    make_gallery(output, selection["selected_ids"], rows)
    atomic_json(output/"render.json", dict(selection_sha256=sha256(output/"selection.json"),
                                          metric_config=config.__dict__, code_sha256=sha256(__file__),
                                          outputs={str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                                                   if p.is_file() and p.name != "render.json"}))
    print(f"Gallery: {output/'index.html'} | selection {fingerprint(selection)}")


if __name__ == "__main__":
    main()
