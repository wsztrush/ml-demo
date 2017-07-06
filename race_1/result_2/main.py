import json

a = [1, 2, 3]

print(json.dumps(a))
b = json.loads(json.dumps(a))

print(b)