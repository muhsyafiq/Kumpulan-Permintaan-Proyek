import os,pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import httplib
from urlparse import urlparse
from bs4 import BeautifulSoup
import urllib2
import time
import unicodedata
import dateparser as date_parser
import httplib
from time import time

def stopwords_bank():
    t = open(os.getcwd()+'/stopwords_bank.txt','rb')
    f = t.readlines()
    words=[]
    t.close()
    for i in f:
        words.append(i.strip('\r\n'))
        
    return words

def tokenizing(doc):
    r = [',','.','/','<','>','?',';',"'","\\",':','"','|',
         '[',']','{','}','~','`','!','@','#','$','%','^','&',
         '*','(',')','-','_','=','+','  ','1','2','3','4','5','6',
         '7','8','9','0']

    for c in r:
        doc = doc.replace(c,' ',doc.count(c))

    return doc.split()

def filtering(words):
    stopword = stopwords_bank()
    for w in stopword:
        while True:
            if w in words:
                words.remove(w)
            else:
                break

    return words

def stemm_stag(words):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    sentence = words[0]
    for w in range(1,len(words)-1):
        sentence = sentence+' '+words[w]

    stem = stemmer.stem(sentence)

    return stem.split()

def text_mining(sentence):
    t = tokenizing(sentence)
    f = filtering(t)
    s = stemm_stag(f)
    return s

def checkUrl(url):
    p = urlparse(url)
    conn = httplib.HTTPConnection(p.netloc)
    conn.request('HEAD', p.path)
    resp = conn.getresponse()
    return resp.status < 400

def kompas_get_indeks_num(url):
    urlFile = urllib2.urlopen(url)
    urlHtml = urlFile.read()
    urlFile.close()

    soup = BeautifulSoup(urlHtml)
    get = []
    for link in soup.find_all('a',href=True):
        href = link['href']
        if href!=None:
            if len(href.split(url))==2:
                get.append(href)

    n=1
    for links in get:
        s = links.split(url)
        if len(s)==2:
            k = s[1]
            if int(k) > n:
                n=int(k)
    
    return n

def kompas_date_web(year,month,day):
    if day<10:
        day = '0'+str(day)
    else:
        day = str(day)

    if month<10:
        month = '0'+str(month)
    else:
        month = str(month)

    hasil = str(year)+'-'+month+'-'+day
    
    return hasil

def kompas_get_news_within_indeks(url,start,last_page,date):
    get_links = []
    length = last_page-start +1
    for i in range(1,length):
        new_url = url+date+'/'+str(i)
        urlFile = urllib2.urlopen(new_url)
        urlHtml = urlFile.read()
        urlFile.close()

        soup = BeautifulSoup(urlHtml)
        for links in soup.find_all('a',href=True):
            href = links['href']
            if href!=None:
                hr = href.split('kompas.com/read/')
                if len(hr)==2:
                    get_links.append(href)

    return [get_links,last_page]

def kompas_get_all_news(url,start,date):
    last_page = kompas_get_indeks_num(url+date+'/')
    links = kompas_get_news_within_indeks(url,start,last_page,date)
    
    return links

def kompas_date_now():
    y = time.localtime()[0]
    m = time.localtime()[1]
    d = time.localtime()[2]
    konv = kompas_date_web(y,m,d)
    
    return konv
    
def update_link_list(old,new,last_page,last_day):
    if last_day!=time.localtime()[2]:
        old = []
        last_day=time.localtime()[2]
        last_page=1
    else:
        for link in new:
            if link not in old:
                old.append(link)
                kompas_crawl_web(link,last_day,['daily'])

    return [old,last_day,last_page]

    
def kompas_daily_update():
    home_url_kompas = 'http://indeks.kompas.com/news/'
    curr_link_list=[]
    last_day=time.localtime()[2]
    last_page=1
    while True:
        try:
            date = kompas_date_now()
            url = home_url_kompas
            if checkUrl(url):
                links = kompas_get_all_news(url,last_page,date)
                update = update_link_list(curr_link_list,links[0],
                                          last_page,last_day)
                curr_link_list=update[0]
                last_day=update[1]
                last_page=update[2]
            else:
                pass
        except:
            pass

def kompas_search_bydate(y,m,d):
    home=os.getcwd()+'/artikel'
    year_dir = home+'/'+str(y)
    month_dir = year_dir+'/'+str(m)
    day_dir = month_dir+'/'+str(d)
    if os.path.exists(day_dir):
        return
    
    home_url_kompas = 'http://indeks.kompas.com/news/'
    konv = kompas_date_web(y,m,d)
    url = home_url_kompas
    try:
        links = kompas_get_all_news(url,0,konv)
    except:
        while True:
            if checkUrl(url):
                break
        links = kompas_get_all_news(url,0,konv)

    if len(links[0])!=0:
        for link in links[0]:
            kompas_crawl_web(link,d,['bydate',y,m])

    
def scraping(txt):
    url = urllib2.urlopen(txt)
    try:
        content = url.read()
    except httplib.IncompleteRead, e:
        httplib.HTTPConnection._http_vsn = 10
        httplib.HTTPConnection._http_vsn_str = 'HTTP/1.0'
        content = url.read()
        httplib.HTTPConnection._http_vsn = 11
        httplib.HTTPConnection._http_vsn_str = 'HTTP/1.1'

    
    soup = BeautifulSoup(content, "lxml")
    return soup

def find_attr(addr):
    t = unicode(addr.find_all('title')[0].get_text())
    t = str(unicodedata.normalize('NFKD', t).encode('ascii','ignore'))
    w = str(addr.find_all('div',{'class':'read__time'})[0].get_text())
    w = date_parser.parse(w.split(' -')[1],languages=['id','en'])
    w = [w.day,w.month,w.year]
    e = str(addr.find_all('div',{'class':'read__author'})[0].get_text())
    if len(e)==0:
        e='Tidak ditemukan'

    return [t,w,e]

def get_web_content(addr):
    txt = unicode(addr.find_all('div',{'class':'read__content'})[0].get_text())
    p = str(unicodedata.normalize('NFKD', txt).encode('ascii','ignore'))+'.'
    unico = ['\n','\u','\r','\t','\p','\a','\u201c','\u201d']
    for i in unico:
        p = p.replace(i,'',txt.count(i))
        
    return p

def kompas_crawl_web(url,d,mode):
    print url
    s = scraping(url)
    r = find_attr(s)
    isi=get_web_content(s)
##    check = banjarmasin_filter(isi)
##    if not check:
##        return
    mining = text_mining(isi)
    data = {'url':url,'title':r[0],'date':r[1],
            'penulis':r[2],'original_content':isi,
            'mining_content':mining}

    if mode[0]=='bydate':
        y = mode[1]
        m = mode[2]
    else:
        y = time.localtime()[0]
        m = time.localtime()[1]

    simpan_hasil_crawl(y,m,d,data,'kompas')
    

def banjarmasin_filter(txt):
    words = txt.split()
    for w in words:
        if w.lower() in ['banjarmasin','banjarbaru']:
            return True

    return False

def simpan_hasil_crawl(y,m,d,data,news):
    home=os.getcwd()+'/artikel'
    year_dir = home+'/'+str(y)
    month_dir = year_dir+'/'+str(m)
    day_dir = month_dir+'/'+str(d)
    news_dir = day_dir+'/'+news

    if not os.path.exists(year_dir):
        os.mkdir(year_dir)
    if not os.path.exists(month_dir):
        os.mkdir(month_dir)
    if not os.path.exists(day_dir):
        os.mkdir(day_dir)
    if not os.path.exists(news_dir):
        os.mkdir(news_dir)

    u = len(os.listdir(news_dir))

    pickle.dump(data,open(news_dir+'/'+news+'_artikel_'+str(u),'wb'))
    print('Database Update: Artikel baru dari '+news+' telah ditambahkan ke dalam database.')



##w = 'pemerintah'
##s = time()
##t = calculate_tfidf(2017,4,29,w)
##print 'Waktu eksekusi TF-IDF: '+str(time()-s)+' kompas.'
##print 'Hasil TF-IDF pencarian kata '+w
##print(' ')
##print t
    
    
    
    
                                   
                                   
    
                    
    
    
            
    
