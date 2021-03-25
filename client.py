import requests
import json
if __name__ == "__main__":
    res = requests.post('http://localhost:8080/predict', json={'input_data': '[[{"speaker": "A","utterance": "로리 새 프로젝트를 시작하는데 로리씨가 도와줬으면 좋겠어."}]]'})
    result = json.loads(res.text)
    print(result)
