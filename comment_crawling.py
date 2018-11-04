# 댓글을 달 빈 리스트를 생성합니다.
List=[] 
# 라이브러리를 로드합니다.
from bs4 import BeautifulSoup 
import requests 
import re 
import sys 
import pprint 

# 네이버 뉴스 url을 입력합니다.
url="http://news.naver.com/main/hotissue/read.nhn?mid=hot&sid1=100&cid=1079165&iid=2780573&oid=001&aid=0010079145&ptype=052" 

oid=url.split("oid=")[1].split("&")[0] 
aid=url.split("aid=")[1] 
page=1     
header = { 
    "User-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36", 
    "referer":url, 
     
}  
while True : 
    c_url="https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&templateId=default_society&pool=cbox5&_callback=jQuery1707138182064460843_1523512042464&lang=ko&country=&objectId=news"+oid+"%2C"+aid+"&categoryId=&pageSize=20&indexSize=10&groupId=&listType=OBJECT&pageType=more&page="+str(page)+"&refresh=false&sort=FAVORITE"  
# 파싱하는 단계입니다.
    r=requests.get(c_url,headers=header) 
    cont=BeautifulSoup(r.content,"html.parser")     
    total_comm=str(cont).split('comment":')[1].split(",")[0] 
   
    match=re.findall('"contents":([^\*]*),"userIdNo"', str(cont)) 
# 댓글을 리스트에 중첩합니다.
    List.append(match) 
# 한번에 댓글이 20개씩 보이기 때문에 한 페이지씩 몽땅 댓글을 긁어 옵니다.
    if int(total_comm) <= ((page) * 20): 
        break 
    else :  
        page+=1


# 여러 리스트들을 하나로 묶어 주는 함수입니다.
def flatten(l): 
    flatList = [] 
    for elem in l: 
        # if an element of a list is a list 
        # iterate over this list and add elements to flatList  
        if type(elem) == list: 
            for e in elem: 
                flatList.append(e) 
        else: 
            flatList.append(elem) 
    return flatList


# 리스트 결과입니다.
flatten(List)
print(List)
