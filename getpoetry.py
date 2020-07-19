import sys, os
import requests
from bs4 import BeautifulSoup
import time
import random

def get_page_text(url):
    lines = []
    URL = 'http://www.famouspoetsandpoems.com{}'.format(url)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find(style="padding-left:14px;padding-top:20px;font-family:Arial;font-size:13px;")

    for line in results:
        if('<br/>' not in str(line)):
            print(str(line).strip())
            lines.append(str(line).strip())

    return lines


def get_urls_for_theme(theme):
    theme_urls = []
    URL = 'http://www.famouspoetsandpoems.com/thematic_poems.html'
    page = requests.get(URL)
    # print(page.content)
    soup = BeautifulSoup(page.content, 'html.parser')
    # results = soup.find(style="padding-left:14px;padding-top:20px;font-family:Arial;font-size:13px;")
    all_hrefs = soup.find_all('a', text=theme)
    for link_a in all_hrefs:
        print(link_a['href'])
        theme_urls.append(link_a)

    return theme_urls

def get_urls_for_collection(url, limit=1000):
    collection_urls = []
    URL = 'http://www.famouspoetsandpoems.com{}'.format(url)
    print(URL)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    all_hrefs = soup.find_all('a', style="font-weight:bold;font-size:16px;color:#3C605B;font-family:Times New Roman;")
    for link_a in all_hrefs[:limit]:
        print(link_a['href'])
        collection_urls.append(link_a['href'])
        time.sleep(random.randint(1, 5))

    return collection_urls


def main(argv):
    all_urls = []
    theme_name = 'Angel Poems'
    theme_urls = get_urls_for_theme(theme_name)
    for theme_url in theme_urls:
        print(theme_url['href'])
        all_urls.extend(get_urls_for_collection(theme_url['href'], 1000))
        time.sleep(random.randint(1, 5))

    for page_url in all_urls:
        print(page_url)
        lines = get_page_text(page_url)
        print(lines)    
        with open('source/lines_all_{}.txt'.format(theme_name), 'a') as f:
            for item in lines:
                f.write("%s\n" % item)



    # get_page_text('http://www.famouspoetsandpoems.com/poets/alan_seeger/poems/157.html')



if __name__ == "__main__":
    main(sys.argv[1:]) 