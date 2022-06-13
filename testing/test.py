import requests

resp = requests.post("https://prepareimage-r5feak6qfq-et.a.run.app/", files={'image': open('./testing/gutdey.jpeg', 'rb')})
x = ''
if (resp == 'compost'):
    x = "organik"
else:
    x = "non organik"
print(resp.json())
print(x)