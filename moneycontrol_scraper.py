#!/usr/bin/env python3
"""
Moneycontrol Scraper - Fixed
"""

import cloudscraper
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

MONEYCONTROL_URL = "https://www.moneycontrol.com/indian-indices"

def scrape_with_cloudscraper():
    print("🌐 Trying cloudscraper...")
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(MONEYCONTROL_URL, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            print("✅ cloudscraper succeeded!")
            return response.text
        else:
            print(f"⚠️ cloudscraper returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ cloudscraper error: {e}")
        return None

def parse_indices_from_html(html):
    indices = []
    indian_keywords = ["Nifty 50", "Bank Nifty", "Nifty IT", "Nifty Pharma", "Nifty FMCG"]
    global_keywords = ["S&P 500", "Dow Jones", "Nasdaq", "FTSE 100", "DAX", "CAC 40", "Nikkei 225", "Hang Seng", "Shanghai"]
    all_names = indian_keywords + global_keywords
    for name in all_names:
        idx = html.find(name)
        if idx == -1:
            continue
        segment = html[idx:idx+800]
        price_match = re.search(r'>([\d,]+\.?\d*)</', segment)
        if not price_match:
            price_match = re.search(r'([\d,]+\.?\d*)', segment)
        change_match = re.search(r'([+-]\d+\.?\d*%)', segment)
        if price_match and change_match:
            price = float(price_match.group(1).replace(',', ''))
            change_pct = float(change_match.group(1).replace('%', '').replace('+', ''))
            indices.append((name, price, change_pct))
    return indices

def scrape_with_selenium():
    print("🚀 Launching Chrome with Selenium...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(MONEYCONTROL_URL)
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(3)
        page_text = driver.find_element(By.TAG_NAME, "body").text
        lines = page_text.split('\n')
        indian_names = ["Nifty 50", "Bank Nifty", "Nifty IT", "Nifty Pharma", "Nifty FMCG"]
        global_names = ["S&P 500", "Dow Jones", "Nasdaq", "FTSE 100", "DAX", "CAC 40", "Nikkei 225", "Hang Seng", "Shanghai"]
        indices_data = []
        for name in indian_names + global_names:
            for i, line in enumerate(lines):
                if name in line:
                    for j in range(i, min(i+5, len(lines))):
                        price_match = re.search(r'([\d,]+\.?\d*)', lines[j])
                        change_match = re.search(r'([+-]\d+\.?\d*%)', lines[j])
                        if price_match and change_match:
                            price = float(price_match.group(1).replace(',', ''))
                            change_pct = float(change_match.group(1).replace('%', '').replace('+', ''))
                            indices_data.append((name, price, change_pct))
                            break
                    break
        seen = set()
        unique = []
        for name, price, pct in indices_data:
            if name not in seen:
                seen.add(name)
                unique.append((name, price, pct))
        if unique:
            print("✅ Selenium succeeded!")
            return unique
        else:
            print("⚠️ Selenium found no data")
            return None
    except Exception as e:
        print(f"❌ Selenium error: {e}")
        return None
    finally:
        if driver:
            driver.quit()
            print("🔒 Chrome closed.")

def color_arrow(pct):
    if pct > 0:
        return f"🔼 {pct:+.2f}%"
    elif pct < 0:
        return f"🔽 {pct:+.2f}%"
    else:
        return f"➖ {pct:+.2f}%"

def print_table(title, data):
    if not data:
        print(f"\n⚠️ No data for {title}")
        return
    print(f"\n📊 {title}")
    print("=" * 70)
    print(f"{'Index':<25} {'Price':>12} {'Change':>20}")
    print("-" * 70)
    for name, price, pct in data:
        arrow = color_arrow(pct)
        if pct > 0:
            arrow = f"\033[92m{arrow}\033[0m"
        elif pct < 0:
            arrow = f"\033[91m{arrow}\033[0m"
        price_str = f"{price:,.2f}" if price != 0 else "N/A"
        print(f"{name:<25} {price_str:>12} {arrow:>20}")
    print("=" * 70)

def main():
    print(f"🟢 Moneycontrol Scraper Started – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    html = scrape_with_cloudscraper()
    if html:
        indices = parse_indices_from_html(html)
        if indices:
            indian_names = ["Nifty 50", "Bank Nifty", "Nifty IT", "Nifty Pharma", "Nifty FMCG"]
            indian = [(n, p, c) for n, p, c in indices if n in indian_names]
            global_data = [(n, p, c) for n, p, c in indices if n not in indian_names]
            if indian:
                print_table("INDIAN INDICES (cloudscraper)", indian)
            if global_data:
                print_table("GLOBAL INDICES (cloudscraper)", global_data)
            if indian or global_data:
                return
    print("\n⚠️ cloudscraper no data, trying Selenium...")
    selenium_data = scrape_with_selenium()
    if selenium_data:
        indian_names = ["Nifty 50", "Bank Nifty", "Nifty IT", "Nifty Pharma", "Nifty FMCG"]
        indian = [(n, p, c) for n, p, c in selenium_data if n in indian_names]
        global_data = [(n, p, c) for n, p, c in selenium_data if n not in indian_names]
        if indian:
            print_table("INDIAN INDICES (Selenium)", indian)
        if global_data:
            print_table("GLOBAL INDICES (Selenium)", global_data)
        return
    print("\n❌ Both methods failed. Use the link:")
    print(f"\n   {MONEYCONTROL_URL}\n")

if __name__ == "__main__":
    main()
    input("\n✅ Press Enter to exit...")
