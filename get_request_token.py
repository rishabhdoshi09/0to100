from kiteconnect import KiteConnect

api_key = "twrhxo5s4lwh6fc6"  # Apni real API key daalein
api_secret = "rpoaou9qku8hcftq289x17tmvupebbun"  # Apna real API secret daalein

kite = KiteConnect(api_key=api_key)
print("Ye URL open karein aur login karein:")
print(kite.login_url())
