import json

def load_json(file_name):
    with open(file_name, "r") as f:
        json_obj = json.loads(f.read())

    return json_obj


def save_json(file_name, json_obj):
    with open(file_name, "w") as f:
        json.dump(json_obj, f, indent=4)
