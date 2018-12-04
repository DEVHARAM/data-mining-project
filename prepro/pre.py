import json
import re
import requests


#띄어쓰기
def spell(sen):
    url = "https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn"
    params = {
        '_callback': 'jQuery112403385437672261493_1543683810401',
        'q': sen,
        'color_blindness': 0,
        '_': 1543683810402
        }
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, params=params, headers=headers).text
    response = response.replace(params['_callback']+'(', '')
    response = response.replace(');', '')
    response = json.loads(response, strict=False)
    
    result_text = response['message']['result']['html']
    result_text = re.sub(r'<\/?.*?>', '', result_text)
    return result_text


if __name__ == '__main__':
    f = open('simple.txt', 'r', encoding='utf8')
    # 파이참 환경에서 밑에 출력파일이 이상하게 나오므로
    # 다른 환경에서 실행 시켜서 나온 출력 파일인 result.txt로 나머지를 실행할 것이다.
    fw = open('do_not_use_this_result.txt', 'w', encoding='utf8')
    lines = f.readlines()
    for line in lines:
        transfer = spell(line)
        fw.write(transfer)
