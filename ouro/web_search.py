"""
Web search functionality for Ouro.
"""
import json
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

from ouro.logger import get_logger
from ouro.config import (
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_PROVIDER,
    MAX_SEARCH_RESULTS
)

logger = get_logger()


class SearchResult:
    """Simple structure to hold search results."""
    
    def __init__(self, title: str, url: str, snippet: str, source: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
    
    def __str__(self) -> str:
        return f"{self.title}\nURL: {self.url}\n{self.snippet}"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source
        }


class WebSearch:
    """Web search functionality for Ouro."""
    
    def __init__(self):
        """Initialize web search with configured provider."""
        self.enabled = ENABLE_WEB_SEARCH
        self.provider = WEB_SEARCH_PROVIDER
        self.max_results = MAX_SEARCH_RESULTS
        self.logger = logger
        
        # Default headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search(self, query: str, num_results: int = None) -> List[SearchResult]:
        """Execute a web search using the configured provider.
        
        Args:
            query: Search query
            num_results: Number of results to return (defaults to config setting)
            
        Returns:
            List of SearchResult objects
        """
        if not self.enabled:
            self.logger.warning("Web search is disabled in configuration")
            return []
        
        # Use specified num_results or fall back to default
        num_results = num_results or self.max_results
        
        # Route to appropriate search provider
        if self.provider == "duckduckgo":
            return self._search_duckduckgo(query, num_results)
        elif self.provider == "google":
            self.logger.warning("Google search is not fully implemented yet")
            return []
        elif self.provider == "bing":
            self.logger.warning("Bing search is not implemented yet")
            return []
        else:
            self.logger.error(f"Unknown search provider: {self.provider}")
            return []
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[SearchResult]:
        """Search DuckDuckGo.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Use DuckDuckGo's API (note: not an official API)
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'no_redirect': '1'
            }
            
            response = requests.get('https://api.duckduckgo.com/', params=params, headers=self.headers)
            
            if response.status_code != 200:
                self.logger.error(f"DuckDuckGo search failed with status code {response.status_code}")
                return []
            
            data = response.json()
            
            # Process and format results
            results = []
            
            # Add abstract if available
            if data.get('Abstract'):
                results.append(SearchResult(
                    title=data.get('Heading', 'Abstract'),
                    url=data.get('AbstractURL', ''),
                    snippet=data['Abstract'],
                    source="DuckDuckGo Abstract"
                ))
            
            # Add related topics
            if data.get('RelatedTopics'):
                for i, topic in enumerate(data['RelatedTopics']):
                    if i >= num_results - len(results):
                        break
                    if 'Text' in topic:
                        # Extract URL if available
                        url = topic.get('FirstURL', '')
                        results.append(SearchResult(
                            title=topic.get('FirstURL', 'Related Topic').split('/')[-1].replace('_', ' '),
                            url=url,
                            snippet=topic['Text'],
                            source="DuckDuckGo Related"
                        ))
            
            return results[:num_results]
        
        except Exception as e:
            self.logger.error(f"Error in DuckDuckGo search: {e}")
            return []
    
    def get_search_results_as_text(self, query: str, num_results: int = None) -> str:
        """Get search results formatted as plain text.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted string with search results
        """
        results = self.search(query, num_results)
        
        if not results:
            return "No search results found."
        
        text_parts = [f"Search results for: {query}"]
        
        for i, result in enumerate(results, 1):
            text_parts.append(f"\n## Result {i}: {result.title}")
            text_parts.append(f"URL: {result.url}")
            text_parts.append(f"{result.snippet}")
            text_parts.append("")  # Empty line between results
        
        return "\n".join(text_parts)


# Singleton instance for easy import
web_search = WebSearch()