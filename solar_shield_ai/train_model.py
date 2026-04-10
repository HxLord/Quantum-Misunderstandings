"""
Train / smoke-test the Helios Guard flare CNN using only the dummy dataset
(no NASA downloads). Same as running ``helios_guard_flare_cnn_colab.py``.

NASA DONKI (real flares metadata): ``python train_model.py --donki-flr``
(set ``NASA_API_KEY`` for higher rate limits than DEMO_KEY).
"""

from helios_guard_flare_cnn_colab import main

if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Helios Guard training (dummy data)")
    p.add_argument("--quick", action="store_true", help="Faster run with shorter synthetic series")
    p.add_argument(
        "--donki-flr",
        action="store_true",
        help="Fetch NASA DONKI solar flares (FLR) and print summary; no training",
    )
    p.add_argument(
        "--donki-days",
        type=int,
        default=14,
        help="Days back from today for --donki-flr date range",
    )
    a = p.parse_args()

    if a.donki_flr:
        from donki_api import (
            default_date_range_last_days,
            fetch_flr,
            filter_flr_by_class_prefix,
            summarize_flr_json,
        )

        start, end = default_date_range_last_days(a.donki_days)
        rows = fetch_flr(start, end)
        print(summarize_flr_json(rows))
        mx = filter_flr_by_class_prefix(rows, ("M", "X"))
        print("M/X class count:", len(mx))
        sys.exit(0)

    main(quick=a.quick)
