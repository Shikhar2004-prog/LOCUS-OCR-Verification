"""Drive the running demo app in headless Chromium: screen-record a real session and save screenshots.

usage: python app/record_app.py [URL]   (default http://localhost:7860)
Output: app/media/app_session.webm and app/media/screens/*.png
"""
import glob, json, os, shutil, sys, time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
MEDIA = os.path.join(HERE, "media")
SHOTS = os.path.join(MEDIA, "screens")
EX = os.path.join(HERE, "examples")


def main():
    os.makedirs(SHOTS, exist_ok=True)
    tmp = os.path.join(MEDIA, "_rec")
    shutil.rmtree(tmp, ignore_errors=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
                            record_video_dir=tmp, record_video_size={"width": 1440, "height": 900})
        pg = ctx.new_page()
        t0 = time.time()
        marks = {}
        mark = lambda k: marks.__setitem__(k, round(time.time() - t0, 2))
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=os.path.join(SHOTS, "00_start.png"))
        for i, doc in enumerate(("invoice", "receipt")):
            pg.locator("#doc_img input[type=file]").set_input_files(os.path.join(EX, f"{doc}.png"))
            pg.wait_for_timeout(1500)
            mark(f"{doc}_click")
            pg.get_by_role("button", name="Read and verify").click()
            pg.get_by_text("flagged for review").first.wait_for(timeout=180000)
            mark(f"{doc}_done")
            pg.wait_for_timeout(1500)
            pg.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})"); pg.wait_for_timeout(2500)
            mark(f"{doc}_top")
            pg.screenshot(path=os.path.join(SHOTS, f"{i + 1:02d}_{doc}_result.png"))
            if doc == "invoice":
                pg.wait_for_timeout(2500)
                mark("invoice_scroll1")
                pg.mouse.wheel(0, 900); pg.wait_for_timeout(2500)
                pg.screenshot(path=os.path.join(SHOTS, "01b_invoice_flagged.png"))
                pg.mouse.wheel(0, 900); pg.wait_for_timeout(2500)
                pg.screenshot(path=os.path.join(SHOTS, "01c_invoice_table.png"))
                pg.mouse.wheel(0, -1800); pg.wait_for_timeout(1500)
        mark("word_tab")
        pg.get_by_role("tab", name="2. Inspect one word").click()
        pg.wait_for_timeout(1500)
        pg.locator("#word_img input[type=file]").set_input_files(os.path.join(EX, "words", "invoice_0.png"))
        pg.get_by_label("What the OCR engine claimed").fill("640")
        pg.wait_for_timeout(1000)
        mark("word_click")
        pg.get_by_role("button", name="Verify this word").click()
        pg.get_by_text("P(correct) =").first.wait_for(timeout=120000)
        pg.wait_for_timeout(1500)
        pg.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})"); pg.wait_for_timeout(2500)
        mark("word_top")
        pg.screenshot(path=os.path.join(SHOTS, "03_word_640.png"))
        pg.mouse.wheel(0, 700); pg.wait_for_timeout(2500)
        pg.screenshot(path=os.path.join(SHOTS, "03b_word_640_graph.png"))
        mark("word_end")
        pg.mouse.wheel(0, -700); pg.wait_for_timeout(1000)
        pg.get_by_role("tab", name="About & results").click()
        pg.wait_for_timeout(2500)
        pg.screenshot(path=os.path.join(SHOTS, "04_about.png"))
        pg.wait_for_timeout(1000)
        ctx.close(); b.close()
    vid = glob.glob(os.path.join(tmp, "*.webm"))[0]
    shutil.move(vid, os.path.join(MEDIA, "app_session.webm"))
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(marks, open(os.path.join(MEDIA, "app_session_marks.json"), "w"), indent=1)
    print(marks)
    print("saved", os.path.join(MEDIA, "app_session.webm"))


if __name__ == "__main__":
    main()
