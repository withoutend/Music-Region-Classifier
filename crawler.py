#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from youtube_dl import YoutubeDL
import requests
from bs4 import BeautifulSoup as bs4
import time
import re
import os


# In[ ]:


def download_mp3(url, path):
    ydl_opts = {'format_spec': 'mp3',
                'outtmpl': path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',}],
    }
    if len(url) > 1:
        ydl = YoutubeDL(ydl_opts)
        print(url)
        ydl.download([url])
    else:
        print("Enter list of urls to download")


# In[ ]:


def download_yb(url):
    download_mp3(url, '%(title)s-%(id)s.%(ext)s')


# In[ ]:


if __name__ == '__main__':
    country_list = [['Taiwan','PLFgquLnL59amN9tYr7o2a60yFUfzQO3sU'],
                   ['United Kingdom','PLFgquLnL59amEA53mO3KiIJRSNAzO-PRZ'],
                   ['Turkey','PLFgquLnL59an-05S-d-D1md6qdfjC0GOO'],
                   ['Thailand','PLFgquLnL59anecQ1woaImBSMJDwfrYjmz'],
                   ['South Korea','PLFgquLnL59alGJcdc0BEZJb2p7IgkL0Oe'],
                   ['Russia','PLFgquLnL59an-oQxF1-GKCJ-0eWXYkOoH'],
                   ['Philippines','PLFgquLnL59anCXm7LbIFMGJVvervbfw_k'],
                   ['Mexico','PLFgquLnL59alW2NIFZN8aD00TCJflQb7J'],
                   ['Japan','PLFgquLnL59alxIWnf4ivu5bjPeHSlsUe9'],
                   ['Indonesia','PLFgquLnL59alQ4PrI-9tZyl0Z8Bqp-RE7'],
                   ['India','PLFgquLnL59alF0GjxEs0V_XFCe7LM3ReH'],
                   ['Germany','PLFgquLnL59alxKOClL2CCGsejK4H9HUCV'],
                   ['France','PLFgquLnL59ak5FwmTB7DRJqX3M2B1D7xI'],
                   ['Brazil','PLFgquLnL59amgHJoypBNANk_038__LaXM'],  
                   ['United States','PLFgquLnL59alW3xmYiWRaoz0oM3H17Lth']
                   ['Poland','PLFgquLnL59alUd_1FFpbZH2mhqowGXAdI']
                  ]
    for country in country_list:
        directory = 'Data/' + country[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        s = requests.Session()
        target = 'https://www.youtube.com/playlist?list=' + country[1]
        base = 'https://www.youtube.com/watch?v='
        r = s.get(target)
        soup = bs4(r.text, 'html.parser')
        post_tag = soup.find('body', attrs={'dir':'ltr'})
        if country[0] == 'United Stats':
            music_tag = soup.find_all('a', attrs={'class':'yt-simple-endpoint style-scope ytd-playlist-panel-video-renderer','id':'wc-endpoint'})     
            #href="/watch?v=x3bfa3DZ8JM&amp;index=1&amp;list=RDCLAK5uy_kmPRjHDECIcuVwnKsx2Ng7fyNgFKWNJFs"
            print("US")
        else:
            music_tag = soup.find_all('a', attrs={'class':'yt-uix-sessionlink','dir':'ltr'})
            
        with open('Data/'+ country[0] + '.csv', 'w', newline='') as csvfile:
            print('link,views',file=csvfile)
            for tag in music_tag:
                try:
                    yt_link = tag['href'].split('&')[0].split('=')[1]
                    print(yt_link)
                    print('---------')
                    download_mp3(yt_link, 'Data/'+ country[0] + '/%(title)s-%(id)s.%(ext)s')
                    time.sleep(3)
                    r = s.get(base + yt_link) 
                    viewsoup = bs4(r.text, 'html.parser')
                    views_tag = viewsoup.find('div', attrs={'class':'watch-view-count'}).text
                    print([yt_link, re.sub("\D", "", views_tag)])        
                    print(yt_link + ',' + re.sub("\D", "", views_tag), file=csvfile)
                    time.sleep(3)
                except:
                    continue

