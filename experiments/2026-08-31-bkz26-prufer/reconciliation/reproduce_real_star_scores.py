#!/usr/bin/env python3
"""Regroup saved analytic scores; do not rerun numerical moment integration."""
from pathlib import Path
from decimal import Decimal, localcontext
import json

def main():
    data = json.loads(Path(__file__).with_name("real-star-scores.json").read_text())
    assert len(data["stars"]) == data["counts"]["stars"] == 288
    assert len({s["id"] for s in data["stars"]}) == 288
    checked = 0
    with localcontext() as ctx:
        ctx.prec = 140
        for group in data["groups"]:
            rows = [s for s in data["stars"] if s["matrix"] == group["matrix"]
                    and s["trajectory"] == group["trajectory"]
                    and (group["phase"] == "all_phases" or str(s["phase"]) == group["phase"])]
            assert len(rows) == group["stars"]
            assert min(s["degree"] for s in rows) == group["degree_min"]
            assert max(s["degree"] for s in rows) == group["degree_max"]
            scores = {law: sum(Decimal(s["scores"][law]) for s in rows)
                      for law in group["sum_scores"]}
            for law, value in scores.items():
                assert abs(value - Decimal(group["sum_scores"][law])) <= max(abs(value), Decimal(1))*Decimal("1e-110")
                ratio = value / scores["gks_ideal_tree"]
                assert abs(ratio - Decimal(group["ratio_to_gks"][law])) <= Decimal("1e-110")
            lower = sum(Decimal(s["scores"]["cast1_ideal_tree"]) < Decimal(s["scores"]["gks_ideal_tree"]) for s in rows)
            tied = sum(Decimal(s["scores"]["cast1_ideal_tree"]) == Decimal(s["scores"]["gks_ideal_tree"]) for s in rows)
            assert [lower, tied, len(rows)-lower-tied] == group["cast_lower_tied_higher"]
            checked += 1
            if group["phase"] == "all_phases" and not group["matrix"].startswith("spielman"):
                print(group["matrix"], group["trajectory"], len(rows),
                      "CAST/GKS", float(scores["cast1_ideal_tree"]/scores["gks_ideal_tree"]))
    assert checked == 30
    print("PASS: 288/288 retained rows; 30/30 group summaries. Saved-score arithmetic only.")

if __name__ == "__main__":
    main()
