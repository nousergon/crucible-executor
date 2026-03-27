"""
Quick IB Gateway connectivity check — operational diagnostic tool.

Not part of the automated trading pipeline. Run manually after
deployment or IB Gateway restart to verify connectivity:

    python executor/connection_test.py
"""

from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

print("Connected:", ib.isConnected())
print(ib.accountSummary())

ib.disconnect()
print("Disconnected")