"""Render a 1080x1350 card to PNG. usage: python3 tools/render.py <card.html> <out.png>"""
import sys

from playwright.sync_api import sync_playwright

src, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    try:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    except Exception:  # noqa: BLE001
        b = p.chromium.launch(args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1080, "height": 1350}, device_scale_factor=1)
    pg.goto("file://" + src)
    pg.wait_for_timeout(300)
    over = pg.evaluate("document.documentElement.scrollHeight > 1350")
    pg.screenshot(path=out, clip={"x": 0, "y": 0, "width": 1080, "height": 1350})
    print("overflow" if over else "ok")
    b.close()
