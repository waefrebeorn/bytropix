#!/usr/bin/env python3
"""
Local web search tool using duckduckgo HTML scraping directly.
No API keys required.
"""

import sys
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
import re

def search_duckduckgo(query, max_results=10):
    """Search using duckduckgo HTML."""
    results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        for result in soup.find_all('a', {'class': 'result__snippet'}):
            if len(results) >= max_results:
                break
            snippet = result.get_text(strip=True)
            # Find the parent result container
            container = result.find_parent('div', {'class': 'result'})
            if container:
                link_elem = container.find('a', {'class': 'result__url'})
                title_elem = container.find('a', {'class': 'result__snippet'})
                if link_elem and title_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': link_elem.get('href', ''),
                        'snippet': snippet
                    })
    except Exception as e:
        print(f"DDG search error: {e}", file=sys.stderr)
    return results

def search_bing(query, max_results=10):
    """Search using Bing HTML."""
    results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        for li in soup.find_all('li', {'class': 'b_algo'}):
            if len(results) >= max_results:
                break
            title_elem = li.find('h2')
            link_elem = li.find('a')
            snippet_elem = li.find('p') or li.find('div', {'class': 'b_caption'})
            
            title = title_elem.get_text(strip=True) if title_elem else ''
            link = link_elem.get('href', '') if link_elem else ''
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
            
            if title and link:
                results.append({
                    'title': title,
                    'url': link,
                    'snippet': snippet
                })
    except Exception as e:
        print(f"Bing search error: {e}", file=sys.stderr)
    return results

def fetch_page(url, max_chars=5000):
    """Fetch and extract text content from a URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove script/style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        return text[:max_chars]
    except Exception as e:
        return f"Error fetching {url}: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: web_search_tool.py <query> [--fetch] [--max-results N] [--json]")
        sys.exit(1)
    
    query = sys.argv[1]
    fetch = '--fetch' in sys.argv
    max_results = 10
    
    for i, arg in enumerate(sys.argv):
        if arg == '--max-results' and i + 1 < len(sys.argv):
            max_results = int(sys.argv[i + 1])
        if arg == '--engine' and i + 1 < len(sys.argv):
            engine = sys.argv[i + 1]
    
    print(f"Searching for: {query}")
    results = search_bing(query, max_results)
    
    if not results:
        print("Trying DuckDuckGo fallback...")
        results = search_duckduckgo(query, max_results)
    
    if not results:
        print("No results found")
        sys.exit(1)
    
    output = []
    for i, r in enumerate(results):
        entry = f"{i+1}. {r['title']}\n   URL: {r['url']}\n   Snippet: {r['snippet'][:300]}"
        if fetch:
            print(f"Fetching {r['url']}...", file=sys.stderr)
            content = fetch_page(r['url'])
            entry += f"\n   Content: {content[:1500]}"
        output.append(entry)
    
    print("\n\n".join(output))
    
    if '--json' in sys.argv:
        print("\n---JSON---")
        print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()