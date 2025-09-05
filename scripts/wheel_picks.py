#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import pathlib
import random

def main():
    parser = argparse.ArgumentParser(description="Generate wheel lottery picks.")
    parser.add_argument("--tickets", type=int, default=5, help="How many tickets to generate")
    parser.add_argument("--numbers", type=int, default=6, help="Numbers per ticket")
    parser.add_argument("--max", type=int, default=49, help="Max number")
    parser.add_argument("--seed", type=str, default="", help="Optional random seed")
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    outdir = pathlib.Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    draws = []
    for i in range(args.tickets):
        pick = sorted(random.sample(range(1, args.max + 1), args.numbers))
        draws.append({"ticket": i + 1, "numbers": pick})

    meta = {
        "generated_at_utc": ts,
        "tickets": args.tickets,
        "numbers_per_ticket": args.numbers,
        "max_number": args.max,
    }

    json_path = outdir / f"wheel_picks_{ts}.json"
    md_path = outdir / f"wheel_picks_{ts}.md"

    json_path.write_text(json.dumps({"meta": meta, "draws": draws}, indent=2), encoding="utf-8")

    lines = [
        "# Wheel Lottery Picks",
        f"- Generated (UTC): {ts}",
        f"- Tickets: {args.tickets}",
        "",
    ]
    for d in draws:
        lines.append(f"- Ticket {d['ticket']}: {', '.join(str(n) for n in d['numbers'])}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path} and {md_path}")

if __name__ == "__main__":
    main()
