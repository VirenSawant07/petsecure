"""Presentation for app.py (PetSecure): styles, the logo, page sections and small HTML components.
Only presentation lives here; the models, class lists and predictions are in app.py."""
import json
from html import escape
from pathlib import Path

EMOTIONS = ["angry", "happy", "sad"]          # display order of the bars; app.py checks it against the model classes
COLOR = {"angry": "#c8553d", "happy": "#2f8f63", "sad": "#3a6fb8"}
TINT = {"angry": "#fbe9e3", "happy": "#e4f3ea", "sad": "#e5edf8"}
EMOJI = {"angry": "😠", "happy": "😊", "sad": "😢"}
TIER = {"High": ("●", "tier-high"), "Moderate": ("◐", "tier-mod"), "Low": ("○", "tier-low")}
IMG = "app/static/img"                          # served by Streamlit static file serving (see .streamlit/config.toml)
CREDITS = json.loads((Path(__file__).resolve().parent / "static" / "img" / "credits.json").read_text(encoding="utf-8"))
ALT = {c["file"]: c["alt"] for c in CREDITS}

NAME, TAGLINE = "PetSecure", "Understand how your dog really feels — from a photo or a bark."

# ------------------------------------------------------------------ logo: a paw print with a heart-shaped pad inside a soft shield
LOGO_SVG = ('<svg class="logo-svg" viewBox="0 0 64 64" role="img" aria-label="PetSecure logo" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M32 4C40 9 48 11 56 11V30C56 45 45 55 32 60C19 55 8 45 8 30V11C16 11 24 9 32 4Z" fill="#1f4d3a"/>'
            '<ellipse cx="21.5" cy="25" rx="3.8" ry="5" transform="rotate(-22 21.5 25)" fill="#f7ebd6"/>'
            '<ellipse cx="28.5" cy="19.5" rx="4" ry="5.2" transform="rotate(-6 28.5 19.5)" fill="#f7ebd6"/>'
            '<ellipse cx="35.5" cy="19.5" rx="4" ry="5.2" transform="rotate(6 35.5 19.5)" fill="#f7ebd6"/>'
            '<ellipse cx="42.5" cy="25" rx="3.8" ry="5" transform="rotate(22 42.5 25)" fill="#f7ebd6"/>'
            '<path d="M32 47C24 41 21 37 21 33.5C21 30.5 23.5 28.5 26.3 28.5C28.6 28.5 30.6 29.8 32 31.8C33.4 29.8 35.4 28.5 '
            '37.7 28.5C40.5 28.5 43 30.5 43 33.5C43 37 40 41 32 47Z" fill="#e0a458"/></svg>')
PAW = ("<svg xmlns='http://www.w3.org/2000/svg' width='440' height='440' viewBox='0 0 440 440'><g fill='%231f4d3a' fill-opacity='0.045'>"
       "<g transform='translate(70 90) rotate(-18)'><ellipse cx='0' cy='0' rx='9' ry='12'/><ellipse cx='20' cy='-8' rx='9' ry='12'/>"
       "<ellipse cx='40' cy='0' rx='9' ry='12'/><ellipse cx='20' cy='28' rx='18' ry='15'/></g>"
       "<g transform='translate(300 300) rotate(24)'><ellipse cx='0' cy='0' rx='9' ry='12'/><ellipse cx='20' cy='-8' rx='9' ry='12'/>"
       "<ellipse cx='40' cy='0' rx='9' ry='12'/><ellipse cx='20' cy='28' rx='18' ry='15'/></g></g></svg>")

CSS = """<style>
:root { --bg:#fbf7f0; --surface:#ffffff; --card:#f5eee3; --ink:#1f2a24; --muted:#5b665f; --line:#ebe2d4;
        --primary:#1f4d3a; --primary-2:#2c6a50; --accent:#e0a458; --accent-ink:#8a5412; --track:#efe7da;
        --shadow:0 10px 30px -12px rgba(46,38,24,.22); --radius:24px; }
.stApp { background-color:var(--bg); background-image:url("data:image/svg+xml;utf8,PAW"); background-attachment:fixed; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stAppDeployButton"] { display:none; }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1180px; }
h1, h2, h3, .serif { font-family:"Fraunces", Georgia, serif !important; letter-spacing:-.01em; color:var(--ink); }
a { color:var(--primary-2); }
:focus-visible, summary:focus-visible, a:focus-visible { outline:3px solid var(--accent) !important; outline-offset:3px; border-radius:8px; }
/* ---------- streamlit widgets ---------- */
.stButton > button { border-radius:999px; padding:.65rem 1.3rem; font-weight:600; transition:transform .18s ease, box-shadow .18s ease, background .18s ease; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 18px -8px rgba(31,77,58,.45); }
.stButton > button[kind="primary"] { background:var(--primary); border-color:var(--primary); color:#fff; }
.stButton > button[kind="primary"]:hover { background:var(--primary-2); border-color:var(--primary-2); }
.stButton > button[kind="secondary"] { background:var(--surface); border:1.5px solid var(--primary); color:var(--primary); }
.stDownloadButton > button { border-radius:999px; border:1.5px solid var(--line); background:var(--surface); font-weight:600; }
[class*="st-key-card_"] { background:var(--surface); border-radius:var(--radius) !important; border:1px solid var(--line) !important;
                          box-shadow:var(--shadow); transition:transform .22s ease, box-shadow .22s ease; }
[class*="st-key-card_mode"]:hover { transform:translateY(-6px); box-shadow:0 22px 40px -18px rgba(46,38,24,.35); }
[data-testid="stFileUploaderDropzone"] { border:2px dashed #d9ccb6; background:#fffdf9; border-radius:18px; transition:border-color .2s, background .2s, transform .2s; }
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploaderDropzone"]:focus-within { border-color:var(--primary-2); background:#f3f8f4; transform:scale(1.01); }
[data-testid="stFileUploaderFile"] { animation:fadeUp .45s ease both; }
[data-testid="stImage"] img { border-radius:18px; }
[data-testid="stMetric"] { background:var(--bg); border:1px solid var(--line); border-radius:16px; padding:.6rem .85rem; }
[data-testid="stSelectbox"] > div > div { border-radius:12px; }
/* ---------- motion ---------- */
@keyframes fadeUp { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:none; } }
.reveal { animation:fadeUp .7s ease both; }
@supports (animation-timeline: view()) { .reveal { animation:fadeUp linear both; animation-timeline:view(); animation-range:entry 0% cover 22%; } }
@media (prefers-reduced-motion: reduce) { *, .reveal { animation:none !important; transition:none !important; } }
/* ---------- nav & shared ---------- */
.nav { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.4rem 0 1.2rem; flex-wrap:wrap; }
.brandmark { display:flex; align-items:center; gap:.55rem; text-decoration:none !important; }
.logo-svg { width:42px; height:42px; flex-shrink:0; }
.wordmark { font-family:"Fraunces", Georgia, serif; font-size:1.55rem; font-weight:700; color:var(--primary); letter-spacing:-.01em; }
.wordmark span { color:var(--accent-ink); }
.navlinks { display:flex; gap:1.3rem; flex-wrap:wrap; } .navlinks a { color:var(--muted); text-decoration:none; font-weight:500; font-size:.95rem; }
.navlinks a:hover { color:var(--primary); }
.eyebrow { font-size:.78rem; letter-spacing:.16em; text-transform:uppercase; color:var(--accent-ink); font-weight:700; }
.section { margin:4.2rem 0 1.4rem; }
.section-head { text-align:center; max-width:40rem; margin:0 auto 1.8rem; }
.section-head h2 { font-size:clamp(1.8rem, 3.5vw, 2.5rem); margin:.3rem 0 .4rem; padding:0; }
.section-head p { color:var(--muted); margin:0; font-size:1.05rem; }
.chip { display:inline-flex; align-items:center; gap:.3rem; font-size:.78rem; font-weight:600; padding:.28rem .7rem; border-radius:999px;
        background:var(--card); color:var(--ink); }
.chip.acc { background:#e4f0e8; color:var(--primary); }
/* ---------- hero ---------- */
.hero h1 { font-size:clamp(2.3rem, 5.2vw, 3.7rem); line-height:1.05; margin:.6rem 0 .9rem; padding:0; }
.hero h1 em { font-style:italic; color:var(--primary); background:linear-gradient(transparent 62%, #f3d7ad 62%); padding:0 .1em; }
.hero p.lead { font-size:1.15rem; color:var(--muted); max-width:34rem; margin:0 0 1.2rem; }
.trust { display:flex; gap:.5rem; flex-wrap:wrap; margin:0 0 1.4rem; }
.collage { position:relative; display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:6px; }
.collage img { width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:28px; box-shadow:var(--shadow); display:block; }
.collage img:nth-child(1) { transform:rotate(-2.5deg); } .collage img:nth-child(2) { transform:translateY(26px) rotate(2deg); }
.collage img:nth-child(3) { transform:rotate(1.5deg); } .collage img:nth-child(4) { transform:translateY(26px) rotate(-2deg); }
.float-badge { position:absolute; left:50%; top:50%; transform:translate(-50%,-40%); background:var(--surface); border-radius:999px;
               padding:.55rem 1rem; box-shadow:var(--shadow); font-weight:700; color:var(--primary); white-space:nowrap; font-size:.95rem; }
/* ---------- mode cards ---------- */
.mode img { width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:18px; display:block; }
.mode h3 { font-size:1.7rem; margin:.9rem 0 .2rem; padding:0; }
.mode p { color:var(--muted); margin:0 0 .8rem; }
.mode .chips { display:flex; gap:.4rem; flex-wrap:wrap; margin-bottom:.8rem; }
@media (min-width: 901px) { .mode p { min-height:3.1em; } .mode .chips { min-height:4.3rem; align-content:flex-start; } }
/* ---------- steps ---------- */
.steps { display:grid; grid-template-columns:repeat(3, 1fr); gap:1.4rem; position:relative; }
.steps::before { content:""; position:absolute; top:34px; left:16%; right:16%; border-top:2px dashed #d9ccb6; z-index:0; }
.step { text-align:center; position:relative; z-index:1; }
.step .ic { width:68px; height:68px; border-radius:50%; background:var(--primary); display:grid; place-items:center; margin:0 auto .8rem;
            box-shadow:0 0 0 8px var(--bg); }
.step .ic svg { width:30px; height:30px; stroke:#fff; fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
.step b { font-family:"Fraunces", Georgia, serif; font-size:1.2rem; color:var(--ink); display:block; }
.step span { color:var(--muted); font-size:.95rem; }
/* ---------- emotions / tips / gallery ---------- */
.grid3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:1.3rem; }
.grid4 { display:grid; grid-template-columns:repeat(4, 1fr); gap:1.1rem; }
.ecard, .tip { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow);
               transition:transform .22s ease, box-shadow .22s ease; }
.ecard:hover, .tip:hover { transform:translateY(-5px); box-shadow:0 22px 40px -18px rgba(46,38,24,.35); }
.ecard img { width:100%; aspect-ratio:4/3; object-fit:cover; display:block; }
.ecard .body { padding:1.1rem 1.25rem 1.3rem; }
.ecard h3 { font-size:1.45rem; margin:0 0 .35rem; padding:0; display:flex; align-items:center; gap:.45rem; }
.ecard p { color:var(--muted); margin:0 0 .7rem; font-size:.96rem; }
.ecard ul { margin:0; padding-left:1.1rem; } .ecard li { color:var(--ink); font-size:.92rem; margin:.15rem 0; }
.ecard .lbl { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; color:var(--accent-ink); margin-bottom:.25rem; }
.tip { padding:1.3rem 1.2rem; }
.tip .ti { width:46px; height:46px; border-radius:14px; background:#f6e6cf; display:grid; place-items:center; font-size:1.4rem; margin-bottom:.7rem; }
.tip b { font-family:"Fraunces", Georgia, serif; font-size:1.12rem; color:var(--ink); display:block; margin-bottom:.3rem; }
.tip span { color:var(--muted); font-size:.93rem; }
.strip { display:flex; gap:14px; overflow-x:auto; scroll-snap-type:x mandatory; padding:.4rem .2rem 1rem; scrollbar-width:thin; }
.strip img { width:220px; height:220px; flex:0 0 auto; object-fit:cover; border-radius:22px; scroll-snap-align:start; box-shadow:var(--shadow);
             transition:transform .22s ease; }
.strip img:hover { transform:translateY(-4px) rotate(-1deg); }
/* ---------- faq ---------- */
.faq details { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:.95rem 1.2rem; margin:.6rem auto; max-width:820px; }
.faq summary { cursor:pointer; font-weight:650; color:var(--ink); list-style:none; display:flex; justify-content:space-between; gap:1rem; }
.faq summary::-webkit-details-marker { display:none; }
.faq summary::after { content:"+"; font-size:1.3rem; line-height:1; color:var(--primary); transition:transform .2s; }
.faq details[open] summary::after { transform:rotate(45deg); }
.faq details p { color:var(--muted); margin:.7rem 0 .1rem; }
/* ---------- footer ---------- */
.footer { margin-top:4.5rem; background:var(--primary); color:#e8efe9; border-radius:28px; padding:2.2rem 2rem 1.6rem; }
.footer .cols { display:grid; grid-template-columns:1.4fr 1fr 1.4fr; gap:2rem; }
.footer .wordmark { color:#fff; } .footer .wordmark span { color:var(--accent); }
.footer p, .footer li { color:#cfdcd3; font-size:.9rem; } .footer h4 { color:#fff; font-size:.8rem; letter-spacing:.14em; text-transform:uppercase; margin:0 0 .6rem; }
.footer ul { list-style:none; padding:0; margin:0; } .footer a { color:#f3d7ad; text-decoration:none; } .footer a:hover { text-decoration:underline; }
.footer .legal { border-top:1px solid rgba(255,255,255,.15); margin-top:1.4rem; padding-top:1rem; font-size:.8rem; color:#b7c7bc; }
/* ---------- analysis pages ---------- */
.page-head { display:flex; align-items:center; gap:1rem; margin:.4rem 0 1.3rem; }
.page-head .badge { width:3.6rem; height:3.6rem; border-radius:18px; display:grid; place-items:center; font-size:1.8rem; background:#e4f0e8; }
.page-head h1 { font-size:2.3rem; margin:0; padding:0; } .page-head p { margin:.15rem 0 0; color:var(--muted); }
.card-title { font-family:"Fraunces", Georgia, serif; font-weight:700; font-size:1.25rem; color:var(--ink); margin:.1rem 0 .6rem; }
.verdict { border-radius:22px; padding:1.3rem 1.4rem; display:flex; align-items:center; gap:1.2rem; margin:.2rem 0 1rem; animation:fadeUp .5s ease both; }
.ring { width:112px; height:112px; border-radius:50%; display:grid; place-items:center; flex-shrink:0; }
.ring-in { width:92px; height:92px; border-radius:50%; display:grid; place-items:center; font-size:2.8rem; background:#fff; }
.v-label { font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); font-weight:600; }
.v-emotion { font-family:"Fraunces", Georgia, serif; font-size:2.8rem; font-weight:700; line-height:1.05; }
.v-meta { display:flex; align-items:center; gap:.5rem; margin-top:.4rem; flex-wrap:wrap; }
.conf { font-variant-numeric:tabular-nums; font-weight:700; color:var(--ink); }
.pill { font-size:.78rem; font-weight:700; padding:.18rem .7rem; border-radius:999px; }
.tier-high { background:#e4f3ea; color:#1d5a3c; } .tier-mod { background:#fbeed6; color:#7a4b00; } .tier-low { background:#f8e3de; color:#8a2c22; }
.bars { display:grid; gap:.55rem; margin:.3rem 0 .9rem; }
.bar-row { display:grid; grid-template-columns:6rem 1fr 3.6rem; align-items:center; gap:.7rem; font-size:.95rem; color:var(--ink); }
.bar-track { height:.75rem; background:var(--track); border-radius:6px; overflow:hidden; }
.bar-fill { height:100%; border-radius:6px; animation:grow .8s ease-out; }
.bar-val { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); font-weight:600; }
@keyframes grow { from { width:0; } }
.gauge-track { display:grid; grid-template-columns:1fr 1fr 1fr; gap:5px; }
.gauge-seg { height:.55rem; border-radius:4px; background:var(--track); }
.gauge-seg.on.Low { background:#d9826d; } .gauge-seg.on.Moderate { background:#e0a458; } .gauge-seg.on.High { background:#2f8f63; }
.gauge-lbl { display:grid; grid-template-columns:1fr 1fr 1fr; font-size:.72rem; color:var(--muted); margin-top:.25rem; text-align:center; }
.fine { font-size:.84rem; color:var(--muted); margin:.35rem 0 0; }
.explain { background:var(--bg); border-radius:18px; padding:1rem 1.1rem; margin:1rem 0 .7rem; }
.explain p { margin:0 0 .4rem; color:var(--ink); } .explain p:last-child { margin:0; }
.care { display:flex; gap:.8rem; align-items:flex-start; background:#fbf1e2; border-radius:18px; padding:1rem 1.1rem; margin:.7rem 0; }
.care .ci { font-size:1.5rem; line-height:1; } .care b { display:block; color:var(--accent-ink); margin-bottom:.15rem; } .care span { color:var(--ink); font-size:.95rem; }
.note { font-size:.84rem; color:var(--muted); margin:.6rem 0; }
.empty { border:2px dashed #d9ccb6; border-radius:20px; padding:2.6rem 1rem; text-align:center; color:var(--muted); font-size:.98rem; background:#fffdf9; }
.empty .big { font-size:2.6rem; display:block; margin-bottom:.4rem; animation:bob 2.4s ease-in-out infinite; }
@keyframes bob { 50% { transform:translateY(-6px); } }
/* ---------- responsive (mobile first overrides) ---------- */
@media (max-width: 900px) { .grid4 { grid-template-columns:repeat(2, 1fr); } .grid3 { grid-template-columns:1fr; max-width:520px; margin:0 auto; }
  .footer .cols { grid-template-columns:1fr; gap:1.2rem; } .navlinks { display:none; } }
@media (max-width: 640px) { .steps { grid-template-columns:1fr; } .steps::before { display:none; } .grid4 { grid-template-columns:1fr; }
  .collage img:nth-child(2), .collage img:nth-child(4) { transform:none; } .v-emotion { font-size:2.2rem; }
  .ring { width:88px; height:88px; } .ring-in { width:72px; height:72px; font-size:2.1rem; } .strip img { width:170px; height:170px; }
  .bar-row { grid-template-columns:5rem 1fr 3.2rem; } }
</style>""".replace("PAW", PAW.replace("<", "%3C").replace(">", "%3E"))

# ------------------------------------------------------------------ content (general dog-behaviour guidance, not veterinary advice)
INFO = {
    "angry": {"means": "The dog feels threatened, guarded or annoyed and may be warning others to back off.",
              "signs": ["Stiff, very still body", "Hard, fixed stare", "Lips pulled back, teeth showing", "Raised hackles", "Low growl or snarl"],
              "care": "Give the dog space and calmly remove whatever is bothering it. Don't stare, reach over it or punish a growl — "
                      "a growl is a useful warning. If it happens often, talk to a vet or a certified behaviourist."},
    "happy": {"means": "The dog is relaxed, comfortable and enjoying the moment.",
              "signs": ["Loose, wiggly body", "Soft eyes", "Relaxed, open mouth", "Tail wagging at mid height", "Play bows and bouncy moves"],
              "care": "A great moment for play, a walk or reward-based training. Notice what made your dog feel this way and do more of it."},
    "sad": {"means": "The dog seems low, lonely, bored or uncomfortable.",
            "signs": ["Head and ears held low", "Avoids eye contact", "Curled up or unusually still", "Whining or soft whimpers",
                      "Less interest in food or play"],
            "care": "Offer calm company, gentle attention and a comfortable spot. If the low mood lasts more than a day or two, or comes "
                    "with changes in eating, sleeping or toilet habits, see a vet."},
}
TIPS = [("🐕", "What a tucked tail means", "A tail tucked between the legs usually signals fear or worry. Give your dog space and remove what scares it."),
        ("🔊", "Why dogs whine", "Whining can mean excitement, a need (toilet, food, attention), anxiety or pain. The situation and body language tell you which."),
        ("〰️", "Not every wag is happy", "A loose, sweeping wag is friendly. A stiff, high, fast wag can mean tension or over-excitement."),
        ("✋", "A growl is a warning", "Growling tells you a dog is uncomfortable. Don't punish it; give space and find the cause.")]
ICONS = {"choose": '<svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="13" rx="3"/><circle cx="12" cy="12.5" r="3.5"/><path d="M8 6l1.5-2h5L16 6"/></svg>',
         "upload": '<svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>',
         "heart": '<svg viewBox="0 0 24 24"><path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.5-7 10-7 10z"/></svg>'}


def img(file: str, cls: str = "", eager: bool = False) -> str:
    return (f'<img src="{IMG}/{file}" alt="{escape(ALT.get(file, "Dog photo"))}" class="{cls}" '
            f'loading="{"eager" if eager else "lazy"}" decoding="async">')


def nav(links: bool = True) -> str:
    items = ('<nav class="navlinks" aria-label="Sections"><a href="#how" target="_self">How it works</a><a href="#emotions" target="_self">Emotions</a>'
             '<a href="#tips" target="_self">Tips</a><a href="#faq" target="_self">FAQ</a></nav>') if links else ""
    return (f'<div class="nav"><a class="brandmark" href="./" target="_self" aria-label="{NAME} home">{LOGO_SVG}'
            f'<span class="wordmark">Pet<span>Secure</span></span></a>{items}</div>')


def hero_text(emotions: list) -> str:
    names = ", ".join(e for e in emotions[:-1]) + f" or {emotions[-1]}"
    return ('<div class="hero reveal"><div class="eyebrow">Dog emotion recognition</div>'
            '<h1>Understand how your dog <em>really</em> feels.</h1>'
            f'<p class="lead">From a photo or a bark, {NAME} reads whether your dog seems {names}, and tells you how sure it is.</p>'
            '<div class="trust"><span class="chip">📷 Photos</span><span class="chip">🎧 Barks &amp; whines</span>'
            f'<span class="chip">🐾 {len(emotions)} emotions</span><span class="chip">🔒 Nothing stored</span></div></div>')


def collage() -> str:
    return ('<div class="collage reveal">' + "".join(img(f"hero-dog-{i}.webp", eager=True) for i in range(1, 5))
            + '<div class="float-badge">🐾 Photo or bark</div></div>')


def section_head(anchor: str, eyebrow: str, title: str, text: str) -> str:
    return (f'<div class="section reveal" id="{anchor}"><div class="section-head"><div class="eyebrow">{escape(eyebrow)}</div>'
            f'<h2>{escape(title)}</h2><p>{escape(text)}</p></div></div>')


def mode(kind: str, title: str, text: str, chips: list, acc: str) -> str:
    c = "".join(f'<span class="chip">{escape(x)}</span>' for x in chips) + f'<span class="chip acc">✓ {escape(acc)}</span>'
    return f'<div class="mode">{img(f"mode-{kind}.webp")}<h3>{escape(title)}</h3><p>{escape(text)}</p><div class="chips">{c}</div></div>'


def steps(emotions: list) -> str:
    items = [("choose", "Choose a mode", "Pick Vision for a photo or Sound for a bark, growl or whine."),
             ("upload", "Add a file", "Upload your own or try one of our examples."),
             ("heart", "Read the mood", f"See {', '.join(emotions)}, how sure {NAME} is, and a care tip.")]
    return ('<div class="steps reveal">' + "".join(f'<div class="step"><div class="ic" aria-hidden="true">{ICONS[k]}</div><b>{t}</b><span>{d}</span></div>'
                                                  for k, t, d in items) + "</div>")


def emotion_cards(emotions: list) -> str:
    cards = "".join(
        f'<article class="ecard reveal">{img(f"emotion-{e}.webp")}<div class="body"><h3>{EMOJI[e]} {e.title()}</h3>'
        f'<p>{escape(INFO[e]["means"])}</p><div class="lbl">Signs to look for</div><ul>'
        + "".join(f"<li>{escape(s)}</li>" for s in INFO[e]["signs"]) + "</ul></div></article>" for e in emotions)
    return f'<div class="grid3">{cards}</div>'


def gallery() -> str:
    return ('<div class="strip reveal" tabindex="0" role="region" aria-label="Gallery of happy dogs, scroll sideways">'
            + "".join(img(f"gallery-{i}.webp") for i in range(1, 9)) + "</div>")


def tips() -> str:
    return '<div class="grid4">' + "".join(f'<div class="tip reveal"><div class="ti" aria-hidden="true">{i}</div><b>{escape(t)}</b><span>{escape(d)}</span></div>'
                                           for i, t, d in TIPS) + "</div>"


def faq(emotions: list, img_acc: str, snd_acc: str) -> str:
    qa = [("Are my photos and recordings stored?",
           f"No. Files are processed in memory while the page is open to make the prediction; {NAME} does not save them. "
           "A report file is only created when you click a download button."),
          (f"How accurate is {NAME}?",
           f"On our datasets, photos are recognised with {img_acc} and sounds with {snd_acc} average accuracy under repeated five-fold "
           "cross-validation. Every result also shows a confidence level, so you know when to double-check."),
          ("Which files can I use?",
           "Photos: JPG or PNG. Sounds: WAV, FLAC, MP3 or OGG (the middle 3 seconds are analysed, and a clip must be at least 0.1 s long). "
           "Files can be up to 200 MB."),
          (f"Which emotions can {NAME} read?", f"Three: {', '.join(emotions)}. It works on dogs; results for other animals are not meaningful."),
          ("Can it replace a vet?", "No. It gives a quick, probabilistic reading. For health or behaviour worries, always talk to a vet or a qualified behaviourist.")]
    return '<div class="faq reveal">' + "".join(f"<details><summary>{escape(q)}</summary><p>{escape(a)}</p></details>" for q, a in qa) + "</div>"


def footer() -> str:
    credits = ", ".join(f'<a href="{c["url"]}" target="_blank" rel="noopener">{escape(c["author"])}</a>'
                        for c in {c["author"]: c for c in CREDITS}.values())
    return (f'<footer class="footer"><div class="cols"><div><div class="brandmark">{LOGO_SVG}<span class="wordmark">Pet<span>Secure</span></span></div>'
            f'<p>{escape(TAGLINE)}</p><p>{NAME} is based on our research on multimodal dog emotion recognition; our survey paper was '
            'presented at ISBM 2026.</p></div>'
            '<div><h4>Explore</h4><ul><li><a href="?page=visual" target="_self">Analyse a photo</a></li><li><a href="?page=audio" target="_self">Analyse a sound</a></li>'
            '<li><a href="./#emotions" target="_self">Emotions</a></li><li><a href="./#faq" target="_self">FAQ</a></li></ul></div>'
            f'<div><h4>Photo credits</h4><p>Dog photos from <a href="https://unsplash.com" target="_blank" rel="noopener">Unsplash</a> by {credits}.</p></div></div>'
            f'<div class="legal">⚠️ {NAME} is not a substitute for a vet. Predictions are probabilistic and support, not replace, the judgement of owners '
            "and veterinarians. · Department of Computer Engineering, JSPM's Rajarshi Shahu College of Engineering, Pune.</div></footer>")


# ------------------------------------------------------------------ analysis pages
def page_head(icon: str, title: str, text: str) -> str:
    return f'<div class="page-head reveal"><div class="badge" aria-hidden="true">{icon}</div><div><h1>{escape(title)}</h1><p>{escape(text)}</p></div></div>'


def empty(icon: str, text: str) -> str:
    return f'<div class="empty" role="status"><span class="big" aria-hidden="true">{icon}</span>{escape(text)}</div>'


def verdict(label: str, p: dict, tier: str) -> str:
    best = max(p, key=p.get); deg = round(p[best] * 360, 1)
    icon, cls = TIER[tier]
    return (f'<div class="verdict" style="background:{TINT[best]}" role="status">'
            f'<div class="ring" role="img" aria-label="{best}, {p[best] * 100:.1f} percent" '
            f'style="background:conic-gradient({COLOR[best]} {deg}deg, #ffffffaa 0deg)"><div class="ring-in">{EMOJI[best]}</div></div>'
            f'<div><div class="v-label">{escape(label)}</div><div class="v-emotion" style="color:{COLOR[best]}">{best.title()}</div>'
            f'<div class="v-meta"><span class="conf">{p[best] * 100:.1f}% sure</span><span class="pill {cls}">{icon} {tier} confidence</span></div></div></div>')


def bars(p: dict) -> str:
    rows = "".join(
        f'<div class="bar-row"><span>{EMOJI[e]} {e.title()}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{p[e] * 100:.1f}%;background:{COLOR[e]}"></div></div>'
        f'<span class="bar-val">{p[e] * 100:.1f}%</span></div>' for e in EMOTIONS)
    return f'<div class="bars" role="list" aria-label="Probability of each emotion">{rows}</div>'


def gauge(tier: str, margin: float, uncertainty: float) -> str:
    segs = "".join(f'<div class="gauge-seg{" on " + tier if i <= ["Low", "Moderate", "High"].index(tier) else ""}"></div>' for i in range(3))
    return (f'<div aria-label="Confidence {tier}"><div class="gauge-track">{segs}</div>'
            '<div class="gauge-lbl"><span>Low</span><span>Moderate</span><span>High</span></div></div>'
            f'<p class="fine">Gap to the runner-up: <b>{margin * 100:.1f} points</b> · uncertainty index <b>{uncertainty:.3f}</b> '
            '(0 = certain, 1 = all emotions equally likely)</p>')


def explain(p: dict, tier: str, verb: str) -> str:
    best = max(p, key=p.get)
    sure = {"High": f"{NAME} is confident about this one.",
            "Moderate": "It's fairly sure, but it's worth checking your dog's body language yourself.",
            "Low": "This is a close call, so treat it as a hint and look at the whole dog and the situation."}[tier]
    return (f'<div class="explain"><p><b>In plain English:</b> {NAME} thinks this dog {verb} <b>{best}</b>. {sure}</p>'
            f'<p>{escape(INFO[best]["means"])}</p></div>'
            f'<div class="care"><span class="ci" aria-hidden="true">💡</span><div><b>Care tip</b><span>{escape(INFO[best]["care"])}</span></div></div>')


def note(text: str) -> str:
    return f'<p class="note">{escape(text)}</p>'
