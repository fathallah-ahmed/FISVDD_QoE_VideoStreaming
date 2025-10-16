import requests

URL = "http://127.0.0.1:8000/score"

normal = {
  "vmaf_mean": 94.0,  # high visual quality
  "vmaf_std":  1.0,   # stable quality
  "vmaf_mad":  0.4,   # tiny changes between seconds
  "bitrate_mean": 3200.0,  # healthy bitrate (kbps)
  "stall_ratio":  0.0,     # no rebuffering in the window
  "tsl_end":      20.0     # long time since last stall
}

anomaly = {
  "vmaf_mean": 38.0,  # poor quality
  "vmaf_std":  9.0,   # very unstable
  "vmaf_mad":  4.5,   # big swings per second
  "bitrate_mean": 300.0,   # very low bitrate
  "stall_ratio":  0.8,     # 80% of the window stalled
  "tsl_end":      1.0      # stall happened very recently
}

for name, payload in [("NORMAL (expected False)", normal),
                      ("ANOMALY (expected True)", anomaly)]:
    r = requests.post(URL, json=payload, timeout=10)
    print(f"{name}: {r.status_code} {r.json()}")