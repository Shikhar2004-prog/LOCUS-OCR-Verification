"""Synthetic, licence-free demo documents with word-level ground truth.

Each document is typeset word by word (so every word's box and true text are known), then
degraded the way real scans and phone photos are (low resolution, blur, sensor noise, faded
ink, JPEG). Tesseract then makes genuine recognition errors on them, which the demo and the
video use to show what LOCUS catches. Output: app/examples/<name>.png and <name>.json
"""
import io, json, os, random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "examples")
F = {"sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "sansb": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
     "monob": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
     "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
     "serifb": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"}


class Doc:
    def __init__(self, W, H, scale=3):
        self.s = scale
        self.img = Image.new("L", (W * scale, H * scale), 255)
        self.d = ImageDraw.Draw(self.img)
        self.words = []

    def line(self, x, y, text, font="sans", size=14, gap=None, right=None):
        f = ImageFont.truetype(F[font], size * self.s)
        sp = f.getlength(" ")
        toks = text.split(" ")
        if right is not None:
            x = right - sum(f.getlength(t) for t in toks) / self.s - sp * (len(toks) - 1) / self.s
        cx = x * self.s
        for t in toks:
            if t:
                l, tp, r, b = self.d.textbbox((cx, y * self.s), t, font=f)
                self.d.text((cx, y * self.s), t, fill=0, font=f)
                self.words.append({"text": t, "box": [l / self.s, tp / self.s, r / self.s, b / self.s]})
            cx += f.getlength(t) + sp

    def rule(self, y, x0=40, x1=None):
        x1 = x1 or self.img.size[0] / self.s - 40
        self.d.line((x0 * self.s, y * self.s, x1 * self.s, y * self.s), fill=90, width=self.s)

    def degrade(self, down=1.0, blur=0.6, noise=10, fade=0.0, jpeg=35, seed=0):
        rng = np.random.RandomState(seed)
        W, H = self.img.size[0] // self.s, self.img.size[1] // self.s
        im = self.img.resize((W, H), Image.LANCZOS)
        if down != 1.0:  # low-resolution capture, then upsampled back by the viewer
            im = im.resize((int(W * down), int(H * down)), Image.BILINEAR).resize((W, H), Image.BILINEAR)
        im = im.filter(ImageFilter.GaussianBlur(blur))
        a = np.asarray(im).astype(np.float32)
        if fade:  # faded thermal/toner ink, uneven across the page
            grad = np.linspace(1 - fade, 1, H)[:, None] * np.ones((1, W))
            a = 255 - (255 - a) * grad
        a = np.clip(a + rng.normal(0, noise, a.shape), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=jpeg)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def invoice():
    d = Doc(820, 1000)
    d.line(40, 40, "NORTHFIELD OFFICE SUPPLIES LTD", "sansb", 20)
    d.line(40, 72, "Unit 14, Riverside Trade Park, Leeds LS11 5QT", "sans", 12)
    d.line(40, 92, "VAT No. GB 284 5531 07   Tel 0113 496 0821", "sans", 12)
    d.line(560, 40, "TAX INVOICE", "sansb", 18)
    d.line(560, 76, "Invoice No: INV-20817", "sans", 12)
    d.line(560, 96, "Date: 14/03/2026", "sans", 12)
    d.line(560, 116, "Due: 13/04/2026", "sans", 12)
    d.line(40, 140, "Bill to:", "sansb", 12)
    d.line(40, 160, "Harbourline Logistics Pvt Ltd", "sans", 12)
    d.line(40, 180, "Accounts Payable, 7 Quay Street, Hull HU1 1TB", "sans", 12)
    d.rule(215)
    for x, t in ((40, "Item"), (110, "Description"), (470, "Qty"), (560, "Unit"), (690, "Amount")):
        d.line(x, 222, t, "sansb", 12)
    d.rule(244)
    items = [("A4-80", "Copier paper A4 80gsm (box of 5)", 12, 21.45), ("TN-247", "Toner cartridge black TN247", 3, 64.90),
             ("EN-C5", "Envelopes C5 self seal (500)", 4, 18.75), ("PN-BLU", "Ballpoint pens blue (50 pack)", 6, 7.35),
             ("LF-ARC", "Lever arch files A4 assorted", 20, 2.89), ("ST-118", "Staples 26/6 (5000)", 10, 1.15),
             ("WB-MKR", "Whiteboard markers set of 8", 5, 6.40), ("LB-L7", "Address labels L7163 (100 sh)", 2, 23.99)]
    y, sub = 254, 0.0
    for code, desc, q, u in items:
        amt = q * u
        sub += amt
        d.line(40, y, code, "sans", 12)
        d.line(110, y, desc, "sans", 12)
        d.line(470, y, str(q), "sans", 12)
        d.line(560, y, f"{u:.2f}", "sans", 12)
        d.line(0, y, f"{amt:,.2f}", "sans", 12, right=770)
        y += 26
    d.rule(y + 4)
    vat = round(sub * 0.2, 2)
    for lab, v in (("Subtotal", sub), ("VAT 20%", vat), ("TOTAL GBP", sub + vat)):
        y += 26
        d.line(530, y, lab, "sansb" if "TOTAL" in lab else "sans", 13)
        d.line(0, y, f"{v:,.2f}", "sansb" if "TOTAL" in lab else "sans", 13, right=770)
    y += 50
    d.line(40, y, "Payment by BACS to Sort Code 40-11-62 Account 31926458", "sans", 12)
    d.line(40, y + 20, "Reference INV-20817 within 30 days. Thank you for your business.", "sans", 12)
    return d, d.degrade(down=0.7, blur=0.55, noise=9, jpeg=38, seed=1)


def receipt():
    d = Doc(420, 820)
    d.line(110, 30, "GREEN BASKET MART", "monob", 18)
    d.line(70, 58, "22 Station Road, Mandi 175005", "mono", 12)
    d.line(110, 76, "GSTIN 02AABCG4417K1Z9", "mono", 12)
    d.line(30, 104, "Bill: 0047731   14-03-2026 18:42", "mono", 12)
    d.line(30, 122, "Cashier: Ritu   Counter 3", "mono", 12)
    d.rule(144, 30, 390)
    items = [("Basmati Rice 5kg", 1, 649.00), ("Toor Dal 1kg", 2, 162.50), ("Sunflower Oil 1L", 2, 148.00),
             ("Milk Toned 500ml", 4, 27.00), ("Brown Bread", 1, 45.00), ("Eggs (12)", 1, 84.00),
             ("Tomato 1kg", 1, 38.50), ("Onion 2kg", 1, 70.00), ("Green Tea 100g", 1, 215.00),
             ("Dish Soap 750ml", 1, 99.00)]
    y, tot = 152, 0.0
    for name, q, u in items:
        tot += q * u
        d.line(30, y, name, "mono", 12)
        d.line(240, y, f"{q}x{u:.2f}", "mono", 12)
        d.line(0, y, f"{q * u:.2f}", "mono", 12, right=390)
        y += 22
    d.rule(y + 4, 30, 390)
    gst = round(tot * 0.05, 2)
    for lab, v in (("Subtotal", tot), ("GST 5%", gst), ("TOTAL Rs", tot + gst), ("Paid UPI", tot + gst)):
        y += 24
        d.line(30, y, lab, "monob" if "TOTAL" in lab else "mono", 13)
        d.line(0, y, f"{v:.2f}", "monob" if "TOTAL" in lab else "mono", 13, right=390)
    y += 44
    d.line(60, y, "Items: 15   Saved Rs 42.00", "mono", 12)
    d.line(70, y + 20, "Thank you! Visit again", "mono", 12)
    return d, d.degrade(down=0.8, blur=0.5, noise=9, fade=0.35, jpeg=38, seed=2)


def letter():
    d = Doc(820, 700)
    d.line(40, 40, "CLAIM FORM  -  Section B: Incident details", "serifb", 16)
    rows = [("Policy number:", "HL-4471-0928-36"), ("Claimant name:", "Meera Kashyap"),
            ("Date of incident:", "02 February 2026"), ("Location:", "NH-154 near Ghatasani bridge"),
            ("Vehicle registration:", "HP 33 C 8174"), ("Estimated damage:", "Rs 48,750"),
            ("Garage:", "Sharma Auto Works, Mandi"), ("Witness phone:", "98160 27453")]
    y = 90
    for k, v in rows:
        d.line(40, y, k, "serifb", 13)
        d.line(280, y, v, "serif", 13)
        d.rule(y + 22, 280, 760)
        y += 40
    d.line(40, y + 10, "Description: rear bumper and left tail lamp damaged when a", "serif", 13)
    d.line(40, y + 32, "goods carrier reversed into the parked vehicle at 07:15 hours.", "serif", 13)
    d.line(40, y + 54, "Police report filed at Sadar station, FIR 0193/2026.", "serif", 13)
    return d, d.degrade(down=0.62, blur=0.6, noise=10, jpeg=35, seed=3)


def match_truth(gt_words, box):
    """Ground-truth word under a detected box: the GT word containing the box centre
    (largest horizontal overlap if several). None when the detection hits no word."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, ov = None, 0
    for g in gt_words:
        x0, y0, x1, y1 = g["box"]
        if y0 - 2 <= cy <= y1 + 2 and x0 - 2 <= cx <= x1 + 2:
            o = min(x1, box[2]) - max(x0, box[0])
            if o > ov:
                best, ov = g, o
    return best


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    random.seed(0)
    for name, fn in (("invoice", invoice), ("receipt", receipt), ("claim_form", letter)):
        doc, img = fn()
        img.save(os.path.join(OUT, f"{name}.png"))
        json.dump({"words": doc.words, "size": img.size}, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
        print(name, img.size, len(doc.words), "words")
