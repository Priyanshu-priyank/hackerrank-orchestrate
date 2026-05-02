import requests, json
url = "https://openrouter.ai/api/v1/models"
r = requests.get(url)
models = [m['id'] for m in r.json()['data'] if ':free' in m['id']]
print(models)
