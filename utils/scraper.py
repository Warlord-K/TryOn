import requests, json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def extract_link_flipkart(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content)
    return soup.find_all("img", {"class": "_2r_T1I _396QI4"})[0]['src']

def extract_link_myntra(url):
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}

    s = requests.Session()
    res = s.get(url, headers=headers, verify=False)

    soup = BeautifulSoup(res.text, "lxml")

    script = None
    for s in soup.find_all("script"):
        if 'pdpData' in s.text:
            script = s.get_text(strip=True)
            break
    data = json.loads(script[script.index('{'):])
    try:
        link = data['pdpData']['colours'][0]['image']
    except TypeError as e:
        link = data['pdpData']["media"]['albums'][0]['images'][0]['imageURL']
    return link

def extract_link_amazon(url, DRIVER_PATH = "E:\Setups\chromedriver_win32\chromedriver.exe" ):
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1200")
    try:
        driver = webdriver.Chrome("chromedriver",options=options)
    except Exception as e:
        driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source,"html5lib")
    return soup.findAll("img", {"class": "a-dynamic-image a-stretch-horizontal"})[0]['src']

def extract_link(url):
    if 'flipkart' in url:
        return extract_link_flipkart(url)
    if 'myntra' in url:
        return extract_link_myntra(url)
    if 'amazon' in url:
        return extract_link_amazon(url)
    return "Link Not from Flipkart, Myntra or Amazon"