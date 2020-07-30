import math
import re
from collections import Counter
import urllib
import requests
import torch
from bs4 import BeautifulSoup
 
def get_results(resp):
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.content, "html.parser")
        results = []
        urls = []
        try:
            soup.find('span',class_='f').decompose()
        except:
            pass
        try:
            for g in soup.find_all('span', class_='st'):
                results.append(g.text)
 
            for ref in soup.find_all('div', class_='r'):
                href_tag = ref.find_all(href=True)[0]['href']
                urls.append(href_tag)
        except:
            return [], []
        
        return results, urls
 
def search(query, num_results, only_credible_sources=False):
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"
    query = query.replace(' ', '%20')
 
    headers = {"user-agent": USER_AGENT}
    URL = "https://google.com/search?q={}&num={}&pws=0".format(query, num_results-2)
    if only_credible_sources:
        URL += "&as_sitesearch=https://en.wikipedia.org/"
    
        resp = requests.get(URL, headers=headers)
        results1,urls1 = get_results(resp)
 
        springer_URL = f'https://google.com/search?q={query}&num=1&pws=0&as_sitesearch=https://www.springer.com/'
        resp = requests.get(springer_URL, headers=headers)
        results2,urls2 = get_results(resp)
 
        research_URL = f'https://google.com/search?q={query}&num=1&pws=0&as_sitesearch=https://www.researchgate.net/'
        resp = requests.get(research_URL, headers=headers)
        results3, urls3 = get_results(resp)
 
        results = results1 + results2 + results3
        urls = urls1 + urls2 + urls3
        return results, urls
    else:
        resp = requests.get(URL, headers=headers)
        return get_results(resp)
 
 
def search2(query, num_results, only_credible_sources=False):
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"
    query = query.replace(' ', '%20')
 
    headers = {"user-agent": USER_AGENT}
    URL = "https://google.com/search?q={}&num={}&pws=0".format(query,num_results)
    if only_credible_sources:
        resp = requests.get(URL, headers=headers)
        credible_urls = []
        credible_results = []
 
        results,urls = get_results(resp)
 
        for url, result in zip(urls,results):
            if ('wikipedia' in url) or ('researchgate' in url) or ('springer' in url) or ('.edu' in url):
                credible_urls.append(url)
                credible_results.append(result)
                if len(credible_urls) == 4:
                    return credible_results, credible_urls
 
        return credible_results, credible_urls
    else:
        if num_results>4:
            num_results = 4
        URL = f"https://google.com/search?q={query}&num={num_results}&pws=0"
        resp = requests.get(URL, headers=headers)
        return get_results(resp)

def process(model, tokenizer, question, context):
    input_ids = tokenizer.encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
 
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)
 
    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1
 
    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a
    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
 
    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = ' '.join(tokens[answer_start:answer_end+1])
    return answer
 
import re
def answering_question(model, tokenizer, question, num_results = 4, only_wikipedia=False):
    results, urls = search2(question, num_results, only_wikipedia)
    if len(results) == 0:
        return 'Sorry! No results found'
    context = ".".join(results)
    context = re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', context)
    # context = ".".join(context.split()[:300])
    answer = process(model,tokenizer,question, context)
    if answer == '[CLS]':
        return 'Sorry! No results found'
    answer = re.sub(r"""\s([!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~](?:\s|$))""", r'\1', answer)
    return answer.replace(' ##','')