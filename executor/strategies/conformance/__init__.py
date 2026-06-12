"""Slot S conformance kit (M0 / config#990) — see README.md in this directory.

Validate any ExitRule implementation with one command:

    pytest executor/strategies/conformance --exit-rule "your_pkg.your_mod:YourRule"

Run without ``--exit-rule`` to certify the stock rules (the reference
implementations) — CI does this on every push.
"""
