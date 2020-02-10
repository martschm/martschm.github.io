
# Web-Scraping using Scrapy (Python)

## Purpose
This project serves to demonstrate web scraping in Python using Scrapy on the example of Reuters News (https://www.reuters.com). URL, publication date, title and text of the news articles are stored in a pandas dataframe

## Required Packages


```python
import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess

# for the sake of readability - turn off logging
import logging, sys
logging.disable(sys.maxsize)
```

## Scrapter Classes
Any number of scraper classes can be defined here

**Reuters News**


```python
class reuters_news(scrapy.Spider):
    
    name = "reuters_news"
    
    def __init__(self, dictionary, nr_pages):
        self.dictionary = dictionary
        self.nr_pages = nr_pages

    def start_requests(self):
        urls = ["https://uk.reuters.com/news/archive/euro-zone-news?view=page&page=" + \
                str(page) + \
                "&pageSize=10" for page in range(1,self.nr_pages+1)]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.get_press_releases)
        
    def get_press_releases(self, response):
        urls_to_follow = response.css('div.story-content').xpath('//a/@href').extract()
        for url in urls_to_follow:
            if url.startswith("/article/"):
                url_extended = "https://uk.reuters.com" + url
                yield scrapy.Request(url=url_extended, callback=self.get_information)

    def get_information(self, response):
        self.dictionary["scraper"].append(self.name)
        self.dictionary["url"].append(response.url)
        self.dictionary["date"].append(response.css("div.ArticleHeader_date ::text").extract())
        self.dictionary["title"].append(response.css("h1.ArticleHeader_headline ::text").extract())
        text_tmp = response.css('div.StandardArticleBody_body').xpath('//p/text()').extract()
        # the first (duration of reading) and the last (authors) list element are not useful
        self.dictionary["text"].append(text_tmp[1:-1])
```

## Auxiliary Functions


```python
def convert_if_list(obj):
    """
    This function converts a list of strings (obj) to string
    Input:
        obj - list of strings
    """
    if isinstance(obj, list):
        return " ".join(obj)
    else:
        return obj
```


```python
def unlist_columns(dct):
    """
    This function applies convert_if_list to all keys in the dictionary
    Input:
        dct - dictionary of scraped data
    """
    tmp = dct.copy()
    tmp["scraper"] = [convert_if_list(sublist) for sublist in tmp["scraper"]]
    tmp["url"]     = [convert_if_list(sublist) for sublist in tmp["url"]]
    tmp["date"]    = [convert_if_list(sublist) for sublist in tmp["date"]]
    tmp["title"]   = [convert_if_list(sublist) for sublist in tmp["title"]]
    tmp["text"]    = [convert_if_list(sublist) for sublist in tmp["text"]]
    return tmp
```

## Main Function - Web Scraping


```python
def crawl(scraping_class, nr_pages):
    """
    This function performs the scraping
    Input:
        scraping_class: list of scrapers
        nr_pages: list of number of pages to scrape
    Returns a dataframe with raw data from the websites
    """
    
    # initialize empty dictionary, where we store the scraping results
    dict_raw = {"scraper":[], 
                "url":[], 
                "date":[], 
                "title":[], 
                "text":[]
    }
        
    # start a scrapy crawler process
    process = CrawlerProcess()
    
    # loop over list of scrapers and add them to the scraping process
    if isinstance(scraping_class, list):
        index_pages = 0
        for scraper in scraping_class:
            process.crawl(scraper, dict_raw, nr_pages[index_pages])
            index_pages += 1
    else:
        process.crawl(scraping_class[0], dict_raw, nr_pages[0])
    
    # start scraping
    process.start()
    
    # unlist list of lists to store results in a pd.DataFrame
    dict_unlisted = unlist_columns(dict_raw)
    
    # convert dict to dataframe 
    df_raw = pd.DataFrame(dict_unlisted).drop_duplicates()
    
    return df_raw
```

## Example


```python
# runs around 20 seconds
df_scraped = crawl([reuters_news], [50])
```


```python
df_scraped.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>scraper</th>
      <th>url</th>
      <th>date</th>
      <th>title</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>reuters_news</td>
      <td>https://uk.reuters.com/article/uk-ireland-econ...</td>
      <td>February 5, 2020 /  12:18 AM / 6 days ago</td>
      <td>Irish consumer sentiment climbs to six-month high</td>
      <td>DUBLIN (Reuters) - Irish consumer sentiment hi...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>reuters_news</td>
      <td>https://uk.reuters.com/article/uk-ireland-elec...</td>
      <td>February 8, 2020 /  10:16 PM / 2 days ago</td>
      <td>Near tie between three main parties in Irish e...</td>
      <td>DUBLIN (Reuters) - An Irish national election ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>reuters_news</td>
      <td>https://uk.reuters.com/article/uk-ecb-banks-bb...</td>
      <td>February 6, 2020 /  9:37 PM / 4 days ago</td>
      <td>ECB's de Guindos says BBVA spying case has no ...</td>
      <td>MADRID (Reuters) - European Central Bank Vicep...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_scraped.shape
```




    (626, 5)


