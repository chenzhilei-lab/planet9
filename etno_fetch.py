#!/usr/bin/env python3
"""Fetch ETNO orbital elements from JPL Small-Body Database API."""

import json
import time
import urllib.request

ETNOS = [
    "90377", "2012 VP113", "2015 TG387", "2013 FT28", "2014 SR349",
    "2013 RF98", "2014 FE72", "2015 RX245", "2010 GB174", "2007 TG422",
    "2010 VZ98", "2015 KG163", "2013 RA109", "2015 BP519", "2013 UH15",
    "2013 SY99", "2014 WB556", "2015 RY245", "2021 RR205",
]

SELECTION = {"a_min": 150, "q_min": 30, "cc_max": 5, "arc_min_opp": 2}

results = []
for name in ETNOS:
    try:
        url = f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={name.replace(' ', '%20')}"
        req = urllib.request.Request(url, headers={"User-Agent": "PlanetNine-Audit/1.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())

        if "orbit" in data and "elements" in data["orbit"]:
            els = {e["name"]: float(e["value"]) for e in data["orbit"]["elements"]}
            a = els.get("a", 0)
            q = els.get("q", 0)
            cc = data["orbit"].get("condition_code", 99)
            n_opp = data["orbit"].get("n_del_obs_used", 0) or 0

            if a > SELECTION["a_min"] and q > SELECTION["q_min"] and int(cc) <= SELECTION["cc_max"]:
                e = els.get("e", 0)
                i = els.get("i", 0)
                om = els.get("om", 0)
                w = els.get("w", 0)
                varpi = (w + om) % 360
                results.append({
                    "name": name, "a": round(a, 1), "q": round(q, 1),
                    "e": round(e, 4), "i": round(i, 1), "varpi": round(varpi, 1),
                    "cc": int(cc)
                })
        time.sleep(0.3)
    except Exception as ex:
        print(f"FAIL: {name} ({ex})")

with open("etno_data.json", "w") as f:
    json.dump({"retrieval_date": "2026-06-13", "source": "JPL SSD API",
               "selection": SELECTION, "count": len(results), "objects": results},
              f, indent=2)

print(f"Retrieved {len(results)} ETNOs. Saved to etno_data.json.")
