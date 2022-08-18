import text_summarization
import enko_transfer
from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
import time
import re
from collections import deque
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template

app = Flask(__name__)
run_with_ngrok(app)

# 데이터 카테고리 정의
category2url = {'경제': '/news/economy'}
# URL 정의
base_URL = 'https://www.investing.com'
# requests의 헤더에 보낼 내용, user-agent 정의
# https://www.useragentstring.com/ 에 접속후 User Agent String explained 에 나오는 스트링을 복사 붙여넣기해서 사용한다.
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.81 Safari/537.36'}
# 페이지 단위로 긁어오면서 저장하기

article_list = deque()

@app.route('/', methods=['GET'])
def index():
    # article_list.clear()
    # 페이지 정의
    page = 1
    request_URL = f"{base_URL}{category2url['경제']}/{page}"

    # 해당 페이지에 get 요청
    res = requests.get(request_URL, headers=headers)
    # 문제가 있다면 1분뒤에 다시 request 요청. 
    while res.status_code != requests.codes.ok:
        time.sleep(1)
        print("다시 시도")
        res = requests.get(request_URL, headers=headers)

    # 문제가 생겼을때 바로 프로그램을 중지하려면 아래의 코드를 사용한다.
    # res.raise_for_status()

    # 한글 깨짐 방지 인코딩 정의
    res.encoding = 'UTF-8'

    # web parser시 속도가 제일 빠른 lxmㅣ 라이브러리를 이용
    soup = BeautifulSoup(res.text, 'lxml')

    # 각 기사마다 링크가 담겨있는 a태그 엘리먼트들을 가져온다.
    section = soup.find("section", attrs={"id": "leftColumn"})

    articles = section.find_all("article", attrs={'class': 'js-article-item'})

    # 서버 첫 구동시
    if len(article_list) == 0: 
        for article in articles:
            text_div = article.find('div', attrs={'class': 'textDiv'})
            if text_div.find('span', attrs={'class': 'sponsoredBadge'}):
                continue
            a = text_div.find('a', attrs={'class': 'title'})
            id = a['href'].split('-')[-1]  
            img_URL = article.find('img', attrs={'class': 'lazyload'})['data-src']
            news_title = a.get_text().strip()
            link = base_URL + a['href']
            article_list.append({'img': img_URL, 'original_title': news_title, 'link': link, 'id': id})
            if len(article_list) == 9:
                break
    # 서버가 구동중일 때
    else:
        first_article_id = article_list[0]['id']
        for article in articles:
            text_div = article.find('div', attrs={'class': 'textDiv'})
            if text_div.find('span', attrs={'class': 'sponsoredBadge'}):
                continue
            a = text_div.find('a', attrs={'class': 'title'})
            id = a['href'].split('-')[-1]  
            img_URL = article.find('img', attrs={'class': 'lazyload'})['data-src']
            news_title = a.get_text().strip()
            link = base_URL + a['href']
            if str(id) != first_article_id:
                article_list.appendleft({'img': img_URL, 'original_title': news_title, 'link': link, 'id': id})
                article_list.pop()
            else:
                break
    
    return render_template('index.html', articles=article_list)

@app.route('/article/<int:article_id>', methods=['GET', 'POST'])
def show_article(article_id):
    # url 제거
    pattern1 = '(http|https|ftp|ftps)\:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?'
    pattern2 = r'\([^)]*\)'
    repl = ""

    # 페이지 매칭
    for i in range(len(article_list)):
        if str(article_id) == str(article_list[i]['id']):
            curr_article_index = i
            break
    else:
        return render_template('404.html')
    
    # 이미 가짜DB에 크롤링한 내용이 들어있다면 패스하고 없다면 크롤링하기
    if 'original_content' not in article_list[curr_article_index]:
        # 해당 기사에 get 요청
        res = requests.get(article_list[curr_article_index]['link'], headers=headers)
        # 문제가 있다면 1분뒤에 다시 request 요청. 
        while res.status_code != requests.codes.ok:
            time.sleep(1)
            print("다시 시도")
            res = requests.get(article_list[curr_article_index]['link'], headers=headers)

        # 한글 깨짐 방지 인코딩 정의
        res.encoding = 'UTF-8'

        # web parser시 속도가 제일 빠른 lxmㅣ 라이브러리를 이용
        soup = BeautifulSoup(res.text, 'lxml')

        # 페이지의 왼쪽 섹션만 가져온다.
        section = soup.find("section", attrs={"id": "leftColumn"})

        big_img_URL = section.find('img', attrs={'id': 'carouselImage'})['src']
        # 기사의 내용을 가져온다.
        content = []
        articlePage = section.find("div", attrs={"class":["WYSIWYG", "articlePage"]})

        for ptag in articlePage.find_all("p"):
            p_text = ptag.get_text()
            # 단락내의 url을 제거
            p_text = re.sub(pattern=pattern1, repl=repl, string=p_text)
            p_text = re.sub(pattern=pattern2, repl=repl, string=p_text)
            content.append(p_text)

        article_list[curr_article_index]['original_content'] = content
        article_list[curr_article_index]['big_img'] = big_img_URL

    if request.method == 'POST':
        #여기서 content, title 수정
        if 'converted_content' not in article_list[curr_article_index]:
            article_list[curr_article_index]['converted_title'] = enko_transfer.translate_title(article_list[curr_article_index]['original_title'])
            article_list[curr_article_index]['summarized_text'] = text_summarization.summarize(article_list[curr_article_index]['original_content'])
            article_list[curr_article_index]['converted_content'] = enko_transfer.translate_text(article_list[curr_article_index]['summarized_text'])
        return render_template('article.html', article=article_list[curr_article_index], request_method='POST')

    return render_template('article.html', article=article_list[curr_article_index], request_method='GET')

app.run()