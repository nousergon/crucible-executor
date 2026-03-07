from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

print("Connected:", ib.isConnected())
print(ib.accountSummary())

ib.disconnect()
print("Disconnected")