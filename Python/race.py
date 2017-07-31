import json

if __name__ == '__main__':
    s = '{"a":"123"}'
    obj = json.loads(s)

    print(isinstance(obj.get('a'), str))
