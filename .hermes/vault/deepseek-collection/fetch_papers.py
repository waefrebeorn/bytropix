import urllib.request, os, sys

papers = [
    ("2412.19437", "DeepSeek-V3"),
    ("2512.02556", "DeepSeek-V3.2"),
    ("2502.11089", "Native-Sparse-Attention"),
    ("2405.04434", "DeepSeek-V2"),
    ("2401.06066", "DeepSeekMoE"),
    ("2408.15664", "Auxiliary-Loss-Free-Load-Balancing"),
    ("2505.09343", "Insights-DeepSeek-V3-Scaling"),
    ("2602.21548", "DualPath"),
    ("2601.07372", "Conditional-Memory"),
    ("2401.02954", "DeepSeek-LLM"),
    ("2401.14196", "DeepSeek-Coder"),
    ("2402.03300", "DeepSeekMath"),
    ("2410.13848", "Janus"),
    ("2405.14333", "DeepSeek-Prover"),
    ("2406.11931", "DeepSeek-Coder-V2"),
    ("2408.14158", "Fire-Flyer-AI-HPC"),
    ("2411.07975", "JanusFlow"),
    ("2412.10302", "DeepSeek-VL2"),
    ("2501.12948", "DeepSeek-R1"),
    ("2504.02495", "Inference-Time-Scaling-Reward"),
    ("2504.21801", "DeepSeek-Prover-V2"),
    ("2510.18234", "DeepSeek-OCR"),
    ("2511.22570", "DeepSeekMath-V2"),
    ("2512.24880", "mHC"),
    ("2601.20552", "DeepSeek-OCR-2"),
    ("2407.01906", "Expert-Stick-to-Last"),
    ("2403.05525", "DeepSeek-VL"),
    ("2501.17811", "Janus-Pro"),
]

for arxiv_id, name in papers:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    fname = f"{arxiv_id}_{name}.pdf"
    if os.path.exists(fname) and os.path.getsize(fname) > 10000:
        print(f"  {name}: exists, skip")
        continue
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req, timeout=30).read()
        with open(fname, 'wb') as f:
            f.write(data)
        print(f"  {name}: {len(data)//1024}KB OK")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")
        # Try arxiv abs page for HTML version
        try:
            html_url = f"https://arxiv.org/abs/{arxiv_id}"
            req2 = urllib.request.Request(html_url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urllib.request.urlopen(req2, timeout=30).read()
            with open(f"{arxiv_id}_{name}.html", 'wb') as f:
                f.write(html)
            print(f"  {name}: saved HTML fallback")
        except:
            pass
