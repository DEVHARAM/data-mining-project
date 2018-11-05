#-*-coding: utf-8 -*-

import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions

wd ="/usr/local/bin/chromedriver"  # 다운 받은 웹드라이버 위치
addr = "https://entertain.naver.com/read?oid=076&aid=0003340251" 
# 크롤링하고자하는 사이트 주소

driver = webdriver.Chrome(wd)
driver.get(addr)

'''
pages = 0 # 한 페이지당 약 20개의 댓글이 표시
try:
    while True: # 댓글 페이지가 몇개인지 모르므로.
        driver.find_element_by_css_selector(".u_cbox_btn_view_comment").click() #댓글 더보기버튼 인식이 아직 안됌..
        time.sleep(1.5)
        print(pages,end=" ")
        print(" ")
        pages+=1
    
except exceptions.ElementNotVisibleException as e: # 페이지 끝
    pass
    
except Exception as e: # 다른 예외 발생시 확인
    print(e)
'''
    
html = driver.page_source
dom = BeautifulSoup(html, "lxml")

# 댓글이 들어있는 페이지 전체 크롤링
comments_raw = dom.find_all("span", {"class" : "u_cbox_contents"}) #댓글내용은 <span class="u_cbox_contents"...에 들어있다

# 댓글의 text만 뽑는다.
comments = [comment.text for comment in comments_raw]
print(comments)
comments[:3]

