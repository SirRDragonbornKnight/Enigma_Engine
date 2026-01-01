#!/usr/bin/env python3
"""
Demo of URL safety features.

Shows how the AI can filter unsafe URLs and content.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from enigma.tools.url_safety import URLSafety, ContentFilter

def demo_url_filtering():
    """Demonstrate URL safety filtering."""
    print("=" * 70)
    print("URL SAFETY DEMO")
    print("=" * 70)
    print()
    
    safety = URLSafety()
    
    test_urls = [
        "https://github.com/python/cpython",
        "https://python.org/downloads",
        "https://malware-site.com/virus.exe",
        "https://example.com/download.exe",
        "https://stackoverflow.com/questions/12345",
        "https://example.com/free-crack-download",
        "https://wikipedia.org/wiki/Python",
    ]
    
    print("Testing URLs for safety:\n")
    for url in test_urls:
        is_safe = safety.is_safe(url)
        is_trusted = safety.is_trusted(url)
        
        status = "✅ SAFE" if is_safe else "⛔ BLOCKED"
        trust = " (Trusted)" if is_trusted else ""
        
        print(f"{status}{trust}: {url}")
    
    print()
    print("-" * 70)
    print("Filtering a list of URLs:")
    print()
    
    safe_urls = safety.filter_urls(test_urls)
    print(f"Original: {len(test_urls)} URLs")
    print(f"Filtered: {len(safe_urls)} URLs (safe)")
    print()
    print("Safe URLs:")
    for url in safe_urls:
        print(f"  - {url}")

def demo_content_filtering():
    """Demonstrate content filtering."""
    print()
    print("=" * 70)
    print("CONTENT FILTERING DEMO")
    print("=" * 70)
    print()
    
    content_filter = ContentFilter()
    
    # Test ad detection
    test_texts = [
        "This is good content about Python programming.",
        "Click here to buy now! Limited time offer!",
        "Learn more about data structures and algorithms.",
        "Subscribe now for exclusive content!",
        "This article explains neural networks clearly.",
        "Advertisement: Download our app today!",
    ]
    
    print("Testing content for ads:\n")
    for text in test_texts:
        is_ad = content_filter.is_ad_content(text)
        status = "🚫 AD" if is_ad else "✓ CLEAN"
        print(f"{status}: {text}")
    
    print()
    print("-" * 70)
    print("Filtering mixed content:\n")
    
    mixed_content = """Welcome to our website!
This is a great article about AI.
Click here to buy now and save 50%!
Machine learning is a subset of AI.
Subscribe to our newsletter for updates!
Neural networks process information like the brain.
Limited time offer - act now!
"""
    
    print("Original content:")
    print(mixed_content)
    print()
    
    filtered = content_filter.filter_content(mixed_content)
    print("Filtered content (ads removed):")
    print(filtered)

if __name__ == "__main__":
    demo_url_filtering()
    print()
    demo_content_filtering()
