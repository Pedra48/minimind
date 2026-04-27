import json

def get_texts(data_path):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= 10000: break # 选10000行测试
            try:
                data = json.loads(line)
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                if contents:
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                continue