from kiteconnect import KiteConnect

API_KEY = "twrhxo5s4lwh6fc6"
API_SECRET = "rpoaou9qku8hcftq289x17tmvupebbun"

kite = KiteConnect(api_key=API_KEY)

print("Open the following URL in your browser:")
print(kite.login_url())
print("\nAfter logging in, copy the request_token from the redirect URL.")

request_token = input("\nPaste the request_token here: ")

session = kite.generate_session(request_token, api_secret=API_SECRET)
access_token = session["access_token"]

print("\nYour new access_token (copy this to .env):")
print(access_token)
