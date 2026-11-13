"""
suggest.py - builds suggestions.json: niche song suggestions per genre group, seeded by
the songs posted in the chat (data.json). Runs on GitHub Actions (weekly + whenever the
phone publishes new data). Only needs `requests`.

Sources, combined and scored:
  - Last.fm   similar tracks of the seed songs, top tracks of the group's detailed genres
  - ListenBrainz  artists that listeners of the seed artists also play (names via MusicBrainz)
  - AI curator    Gemini (free tier), Groq as backup - niche picks with a one-line reason
Every pick is checked to exist (Spotify search, or Last.fm as fallback); invented or
unfindable songs are dropped. Suggestions stay fixed within a week, change every week,
and anything that later gets posted in the chat moves to "Picked up".

Secrets (GitHub > Settings > Secrets and variables > Actions):
  LASTFM_API_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, GEMINI_API_KEY, GROQ_API_KEY
All optional except LASTFM_API_KEY; whatever is missing is skipped.
"""

import base64
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(os.environ.get("MUZAK_ROOT", "."))
CACHE_FILE = Path(os.environ.get("MUZAK_CACHE", ".suggest-cache/cache.json"))
PICKS_PER_GROUP = 10
MIN_SONGS_IN_GROUP = 8
SEEDS_PER_GROUP = 12
AI_PICKS_ASKED = 10
AI_SLOTS = 4                     # picks per group reserved for the AI curator (if it has good ones)
MAX_LISTENERS = 1_500_000        # Last.fm listeners above this = too mainstream, dropped
NICHE_SCALE = 300_000            # listeners at which a pick's score is halved
WEEKS_BEFORE_REPEAT = 3
TAG_MAX_POINTS = 0.35            # genre top-tracks: points at the "sweet spot" (rank 50)
POPULAR_ARTIST = 1_000_000       # Last.fm listeners from which an artist counts as very popular
POPULAR_BAND_FACTOR = 0.6        # their top-40% most played recordings score x0.6 (not removed)
RUN_BUDGET_MIN = 30              # after this many minutes, remaining groups use Last.fm only (no timeout)
AI_TIMEOUT = 120                 # seconds to wait for one AI answer
MB_FALLBACK_MAX = 40             # max one-by-one MusicBrainz name lookups per run (1/second)
START = time.time()
UA = "MuzakSuggestions/1.0 (+https://github.com/%s)" % os.environ.get("GITHUB_REPOSITORY", "muzak")
LOG = []


def log(*a):
    msg = " ".join(str(x) for x in a)
    LOG.append(msg)
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def norm(text):
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = re.sub(r"\s-\s.*$", " ", text)
    text = re.sub(r"\b(feat|ft)\.?\b.*$", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def key(title, artist):
    return f"{norm(title)}|{norm(artist)}"


def load(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save(path, data, compact=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=None if compact else 1,
                                     separators=(",", ":") if compact else None), encoding="utf-8")


CACHE = {}


def cached(bucket, k, days, fn):
    """Cache API answers between runs (stored by the workflow's cache step)."""
    b = CACHE.setdefault(bucket, {})
    hit = b.get(k)
    now = time.time()
    if hit and now - hit["t"] < days * 86400:
        return hit["v"]
    v = fn()
    if v is not None:
        b[k] = {"t": now, "v": v}
    return v


def http(method, url, retries=3, **kw):
    kw.setdefault("timeout", 30)
    headers = kw.pop("headers", {}) or {}
    headers.setdefault("User-Agent", UA)
    for attempt in range(retries + 1):
        try:
            r = requests.request(method, url, headers=headers, **kw)
        except requests.RequestException as e:
            if attempt == retries:
                log("  network error:", url.split("?")[0], e.__class__.__name__)
                return None
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", "0") or 0) or 3 * (attempt + 1)
            if attempt == retries:
                return r
            time.sleep(min(wait, 30))
            continue
        return r
    return None


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
def build_groups(data):
    songs = data.get("songs", [])
    known = Counter(s["c"] for s in songs if s.get("c") and s["c"] != "Unknown")
    named = [c for c, _ in known.most_common(7)]
    groups = [c for c, n in known.most_common() if n >= MIN_SONGS_IN_GROUP]
    return songs, groups, named


def seed_weight(s, today):
    try:
        at = datetime.fromisoformat(s["at"][:16])
        age = max((today - at).days, 0)
    except (KeyError, ValueError):
        age = 400
    recency = 1.0 + 0.5 * math.exp(-age / 365)           # recent posts count more: 1.5 (today) -> 1.0 (oldest)
    reshare = 1 + 0.5 * max(s.get("n", 1) - 1, 0)        # songs posted by several people count more
    return round(recency * reshare, 3)


def pick_seeds(group_songs, today):
    """Top songs by weight, spread across posters so one person doesn't dominate."""
    ranked = sorted(group_songs, key=lambda s: -seed_weight(s, today))
    by_poster = defaultdict(list)
    for s in ranked:
        by_poster[s.get("by", "")].append(s)
    seeds, round_ = [], 0
    while len(seeds) < SEEDS_PER_GROUP and any(len(v) > round_ for v in by_poster.values()):
        for v in sorted(by_poster.values(), key=lambda v: -seed_weight(v[0], today)):
            if len(v) > round_ and len(seeds) < SEEDS_PER_GROUP:
                seeds.append(v[round_])
        round_ += 1
    return seeds


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
LASTFM_KEY = os.environ.get("LASTFM_API_KEY", "")


def lastfm(method, **params):
    if not LASTFM_KEY:
        return {}
    r = http("GET", "https://ws.audioscrobbler.com/2.0/", params={
        "method": method, "api_key": LASTFM_KEY, "format": "json", "autocorrect": 1, **params})
    time.sleep(0.2)
    try:
        return r.json() if r is not None else {}
    except ValueError:
        return {}


def lastfm_similar(seed):
    def fetch():
        d = lastfm("track.getsimilar", artist=seed["a"][0], track=seed["t"], limit=30)
        out = []
        for t in (d.get("similartracks") or {}).get("track", []) or []:
            try:
                out.append([t["name"], t["artist"]["name"], float(t.get("match") or 0)])
            except (KeyError, TypeError, ValueError):
                pass
        return out
    return cached("lfm_similar", f"{seed['a'][0]}|{seed['t']}", 30, fetch) or []


def tag_rank_points(rank):
    """Continuous score for a genre's top tracks (rank 1 = biggest hit). The biggest hits still
    count, at half weight; the score rises to its peak at rank 50 and eases off slowly after:
    rank 1 -> 0.18, 25 -> 0.26, 50 -> 0.35, 100 -> 0.28, 150 -> 0.21."""
    if rank <= 50:
        return TAG_MAX_POINTS * (0.5 + 0.5 * rank / 50)
    return TAG_MAX_POINTS * max(1 - 0.4 * (rank - 50) / 100, 0.4)


def lastfm_tag_tracks(tag):
    """A tag's top 150 tracks, in rank order (hits included - scored by tag_rank_points)."""
    def fetch():
        out = []
        for page in (1, 2, 3):
            d = lastfm("tag.gettoptracks", tag=tag, limit=50, page=page)
            for t in (d.get("tracks") or {}).get("track", []) or []:
                try:
                    out.append([t["name"], t["artist"]["name"]])
                except (KeyError, TypeError):
                    pass
        return out
    return cached("lfm_tag150", tag, 21, fetch) or []


def lastfm_listeners(title, artist):
    def fetch():
        d = (lastfm("track.getinfo", artist=artist, track=title).get("track")) or {}
        if not d:
            return -1
        try:
            return int(d.get("listeners") or 0)
        except ValueError:
            return 0
    return cached("lfm_listeners", key(title, artist), 60, fetch)


def artist_info(name):
    def fetch():
        d = lastfm("artist.getinfo", artist=name).get("artist") or {}
        try:
            listeners = int((d.get("stats") or {}).get("listeners") or 0)
        except ValueError:
            listeners = 0
        return {"mbid": d.get("mbid") or "", "listeners": listeners}
    return cached("artist_info", norm(name), 60, fetch) or {"mbid": "", "listeners": 0}


def artist_mbid(name):
    return artist_info(name)["mbid"]


_mb_used = [0]


def _name_from_meta(v):
    """Find a recording's title in a ListenBrainz metadata entry (field names vary)."""
    if not isinstance(v, dict):
        return None
    rec = v.get("recording") if isinstance(v.get("recording"), dict) else {}
    for cand in (rec.get("name"), rec.get("title"), v.get("recording_name"), v.get("name"), v.get("title")):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return None


def recording_titles(mbids):
    """Titles for many recordings at once (ListenBrainz batch lookup, cached). Any it can't name
    fall back to MusicBrainz one by one (1/second), capped per run to keep the job fast."""
    b = CACHE.setdefault("rec_title", {})
    out = {m: b[m]["v"] for m in mbids if m in b}
    todo = [m for m in mbids if m not in out]
    for i in range(0, len(todo), 25):
        chunk = todo[i:i + 25]
        r = http("GET", "https://api.listenbrainz.org/1/metadata/recording/",
                 params={"recording_mbids": ",".join(chunk)}, retries=1)
        body = {}
        try:
            body = r.json() if r is not None and r.ok else {}
        except ValueError:
            pass
        items = body.items() if isinstance(body, dict) else \
            ((x.get("recording_mbid"), x) for x in body if isinstance(x, dict))
        for m, v in items:
            name = _name_from_meta(v)
            if m in chunk and name:
                out[m] = name
                b[m] = {"t": time.time(), "v": name}
    for m in todo:
        if m in out or _mb_used[0] >= MB_FALLBACK_MAX:
            continue
        _mb_used[0] += 1
        r = http("GET", f"https://musicbrainz.org/ws/2/recording/{m}", params={"fmt": "json"}, retries=1)
        time.sleep(1.1)                                   # MusicBrainz asks for max 1 request/second
        try:
            name = (r.json() or {}).get("title") if r is not None and r.ok else None
        except ValueError:
            name = None
        if name:
            out[m] = name
            b[m] = {"t": time.time(), "v": name}
    return out


def listenbrainz_similar(seed_artist):
    """Recordings by artists that ListenBrainz listeners of the seed artist also play, across ALL
    popularity levels, tagged by band: 'top' = the artist's 40% most-played, 'rest' = the other 60%.
    Returns [title, artist, band, weight]: weight is 1.0, except x POPULAR_BAND_FACTOR for the
    'top' band of very popular artists - so their hits still count, just a little less."""
    mbid = artist_mbid(seed_artist)
    if not mbid:
        return []

    def fetch(pop_begin, pop_end):
        r = http("GET", f"https://api.listenbrainz.org/1/lb-radio/artist/{mbid}", params={
            "mode": "medium", "max_similar_artists": 6, "max_recordings_per_artist": 2,
            "pop_begin": pop_begin, "pop_end": pop_end})
        if r is None or not r.ok:
            return []
        try:
            body = r.json()
        except ValueError:
            return []
        entries = []

        def walk(x):
            if isinstance(x, dict):
                if "recording_mbid" in x and "similar_artist_name" in x:
                    entries.append([x["recording_mbid"], x["similar_artist_name"], x.get("similar_artist_mbid", "")])
                else:
                    for v in x.values():
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(body)
        return [e for e in entries if e[2] != mbid]

    bands = [("rest", cached("lb_rest", mbid, 30, lambda: fetch(0, 60)) or []),
             ("top", cached("lb_top", mbid, 30, lambda: fetch(60, 100)) or [])]
    picked = [(band, e) for band, entries in bands for e in entries[:6]]
    titles = recording_titles(list(dict.fromkeys(e[0] for _, e in picked)))
    out, seen = [], set()
    for band, (rec, artist, _) in picked:
            if rec in seen:
                continue
            seen.add(rec)
            title = titles.get(rec)
            if not title:
                continue
            weight = 1.0
            if band == "top" and artist_info(artist)["listeners"] >= POPULAR_ARTIST:
                weight = POPULAR_BAND_FACTOR
            out.append([title, artist, band, weight])
    return out


# ---------------------------------------------------------------------------
# AI curator: Gemini first, Groq as backup
# ---------------------------------------------------------------------------
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
_models = {}


def _json_from_text(text):
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except ValueError:
            continue
    return None


def gemini_models():
    if "gemini" not in _models:
        names = []
        if os.environ.get("GEMINI_MODEL"):
            names.append(os.environ["GEMINI_MODEL"])
        r = http("GET", "https://generativelanguage.googleapis.com/v1beta/models",
                 params={"pageSize": 200}, headers={"x-goog-api-key": GEMINI_KEY}, retries=1)
        found = []
        try:
            for m in (r.json().get("models", []) if r is not None and r.ok else []):
                n = m.get("name", "").replace("models/", "")
                if ("flash" in n and "generateContent" in (m.get("supportedGenerationMethods") or [])
                        and not any(x in n for x in ("image", "tts", "live", "audio", "embedding", "thinking"))):
                    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)[:1]] or [0]
                    found.append((("lite" in n), ("preview" in n or "exp" in n), -nums[0], n))
        except ValueError:
            pass
        names += [n for *_, n in sorted(found)]
        names += ["gemini-flash-latest", "gemini-3.8-flash", "gemini-2.5-flash"]
        _models["gemini"] = list(dict.fromkeys(names))
    return _models["gemini"]


_failed = Counter()


def ask_gemini(prompt):
    if sum(_failed.values()) >= 3:                     # Gemini keeps failing this run: go straight to Groq
        return None, None
    tried = 0
    for model in gemini_models():
        if _failed[model] >= 2 or tried >= 2:        # skip models that keep timing out; max 2 per group
            continue
        tried += 1
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9}}
        r = http("POST", url, json=body, headers={"x-goog-api-key": GEMINI_KEY}, retries=0, timeout=AI_TIMEOUT)
        if r is not None and r.status_code == 400 and "responseMimeType" in r.text:
            body.pop("generationConfig")
            r = http("POST", url, json=body, headers={"x-goog-api-key": GEMINI_KEY}, retries=0, timeout=AI_TIMEOUT)
        if r is None or not r.ok:
            _failed[model] += 1
            continue
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            got = _json_from_text("".join(p.get("text", "") for p in parts if not p.get("thought")))
        except (KeyError, IndexError, ValueError, TypeError):
            got = None
        if got:
            return got, f"Gemini ({model})"
        _failed[model] += 1
    return None, None


def groq_model():
    if "groq" not in _models:
        r = http("GET", "https://api.groq.com/openai/v1/models",
                 headers={"Authorization": f"Bearer {GROQ_KEY}"}, retries=1)
        ids = []
        try:
            ids = [m["id"] for m in (r.json().get("data", []) if r is not None and r.ok else [])]
        except (ValueError, KeyError):
            pass
        prefer = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct",
                  "meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b"]
        chat_ids = [i for i in ids if not any(x in i for x in ("whisper", "tts", "guard", "embed"))]
        _models["groq"] = next((p for p in prefer if p in ids), chat_ids[0] if chat_ids else "llama-3.3-70b-versatile")
    return _models["groq"]


def ask_groq(prompt):
    model = groq_model()
    r = http("POST", "https://api.groq.com/openai/v1/chat/completions",
             headers={"Authorization": f"Bearer {GROQ_KEY}"}, retries=1, timeout=AI_TIMEOUT,
             json={"model": model, "temperature": 0.8, "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": "You are an expert music curator. Reply with JSON only."},
                                {"role": "user", "content": prompt}]})
    if r is None or not r.ok:
        return None, None
    try:
        return _json_from_text(r.json()["choices"][0]["message"]["content"]), f"Groq ({model})"
    except (KeyError, IndexError, ValueError, TypeError):
        return None, None


def ai_picks(group, seeds, top_genres, avoid_artists):
    if not (GEMINI_KEY or GROQ_KEY):
        return [], None
    seed_lines = "\n".join(f"- {s['t']} — {', '.join(s['a'][:2])}" for s in seeds)
    prompt = f"""You are a crate-digging music curator for a group of friends who share music in a chat.
Their taste in the "{group}" part of their playlist:
- Detailed genres they post most: {', '.join(top_genres)}
- Songs they love most in this group:
{seed_lines}

Suggest {AI_PICKS_ASKED} real, released songs they would love that fit this taste but are NICHE:
lesser-known artists, deep cuts, small labels, regional scenes. Avoid mainstream hits.
Do NOT suggest any song by these artists already in their playlist: {', '.join(sorted(avoid_artists)[:120])}.
About two thirds should be artists new to them; the rest can be deeper cuts by related artists.
Only suggest songs you are confident exist, with the exact song title and main artist name.

Reply as JSON only, in this shape:
{{"picks": [{{"title": "...", "artist": "...", "why": "max 18 words: what it shares with their taste, naming a seed song or artist"}}]}}"""
    got, source = (ask_gemini(prompt) if GEMINI_KEY else (None, None))
    if not got and GROQ_KEY:
        got, source = ask_groq(prompt)
    if not got:
        return [], None
    picks = got.get("picks", got) if isinstance(got, dict) else got
    out = []
    for p in picks if isinstance(picks, list) else []:
        if isinstance(p, dict) and p.get("title") and p.get("artist"):
            out.append([str(p["title"])[:120], str(p["artist"])[:120], str(p.get("why", ""))[:200]])
    return out, source


# ---------------------------------------------------------------------------
# Spotify check (client credentials - no user login needed)
# ---------------------------------------------------------------------------
_spotify = {"token": None, "ok": None}


def spotify_token():
    if _spotify["ok"] is None:
        cid, sec = os.environ.get("SPOTIFY_CLIENT_ID", ""), os.environ.get("SPOTIFY_CLIENT_SECRET", "")
        _spotify["ok"] = False
        if cid and sec:
            r = http("POST", "https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"},
                     headers={"Authorization": "Basic " + base64.b64encode(f"{cid}:{sec}".encode()).decode()})
            try:
                if r is not None and r.ok:
                    _spotify["token"], _spotify["ok"] = r.json()["access_token"], True
            except (ValueError, KeyError):
                pass
        if not _spotify["ok"]:
            log("  Spotify check not available - using Last.fm to confirm songs exist instead.")
    return _spotify["token"] if _spotify["ok"] else None


def spotify_find(title, artist):
    token = spotify_token()
    if not token:
        return None

    def fetch():
        q = f'track:"{title}" artist:"{artist}"'
        r = http("GET", "https://api.spotify.com/v1/search", params={"q": q, "type": "track", "limit": 5},
                 headers={"Authorization": f"Bearer {token}"})
        time.sleep(0.15)
        if r is None or not r.ok:
            return None
        items = ((r.json() or {}).get("tracks") or {}).get("items") or []
        nt, na = norm(title), norm(artist)
        for t in items:
            names = [norm(a.get("name")) for a in t.get("artists", [])]
            tn = norm(t.get("name"))
            if na in names and tn and (tn.startswith(nt) or nt.startswith(tn)):
                album = t.get("album") or {}
                imgs = sorted(album.get("images") or [], key=lambda i: i.get("width") or 0)
                img = next((i["url"] for i in imgs if (i.get("width") or 0) >= 160), imgs[-1]["url"] if imgs else "")
                rel = album.get("release_date") or ""
                return {"id": t["id"], "t": t["name"], "a": [a["name"] for a in t.get("artists", [])],
                        "al": album.get("name", ""), "y": int(rel[:4]) if rel[:4].isdigit() else None, "img": img}
        return {}
    return cached("spotify", key(title, artist), 60, fetch)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global CACHE
    data = load(ROOT / "data.json", None)
    if not data:
        sys.exit("data.json not found - publish the site from the phone first.")
    CACHE = load(CACHE_FILE, {})
    prev = load(ROOT / "suggestions.json", {})
    history = load(ROOT / "suggest_history.json", {})
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    week = now.strftime("%G-W%V")
    songs, groups, named = build_groups(data)
    log(f"{len(songs)} songs, {len(groups)} genre groups with {MIN_SONGS_IN_GROUP}+ songs. Week {week}.")
    log("Sources: Last.fm" + (" ✓" if LASTFM_KEY else " ✗ (no key)") + ", ListenBrainz ✓, AI: " +
        ("Gemini" if GEMINI_KEY else "") + (" + Groq backup" if GROQ_KEY else "") + ("none" if not (GEMINI_KEY or GROQ_KEY) else ""))

    in_playlist_ids = {s["id"] for s in songs}
    posted_ids = {p[2] for p in data.get("posts", [])}
    in_playlist_keys = {key(s["t"], a) for s in songs for a in s["a"][:1]}
    playlist_artists = {norm(a) for s in songs for a in s["a"]}
    playlist_artist_names = {a for s in songs for a in s["a"]}
    recent_weeks = {(now - timedelta(weeks=i)).strftime("%G-W%V") for i in range(1, WEEKS_BEFORE_REPEAT + 1)}
    shown_recently = {k for k, h in history.items() if set(h.get("weeks", [])) & recent_weeks}
    shown_recently_ids = {h["id"] for h in history.values() if h.get("id") and set(h.get("weeks", [])) & recent_weeks}
    used_ids = set()
    same_week = prev.get("week") == week
    used_global = set()
    out_groups, ai_used = [], None

    for gi, group in enumerate(groups):
        late = (time.time() - START) / 60 > RUN_BUDGET_MIN
        if late and (gi == 0 or not out_groups[-1].get("quick")):
            log(f"\n(Over {RUN_BUDGET_MIN} min - remaining groups use Last.fm only this run.)")
        gsongs = [s for s in songs if s.get("c") == group]
        seeds = pick_seeds(gsongs, now)
        top_genres = [g for g, _ in Counter(g for s in gsongs for g in s.get("g", [])).most_common(6)]
        log(f"\n[{group}] {len(gsongs)} songs, seeds: {len(seeds)}, genres: {', '.join(top_genres[:4])}")
        cands = {}

        def add(title, artist, score, source, why, seed=None):
            if not title or not artist:
                return
            k = key(title, artist)
            if not k.strip("|") or k in in_playlist_keys:
                return
            c = cands.setdefault(k, {"t": title, "a": artist, "score": 0.0, "src": set(), "why": [], "seeds": []})
            c["score"] += score
            c["src"].add(source)
            if why:
                c["why"].append((source, why))
            if seed and seed not in c["seeds"]:
                c["seeds"].append(seed)

        for s in seeds:
            w = seed_weight(s, now)
            for title, artist, match in lastfm_similar(s)[:25]:
                add(title, artist, w * match, "Last.fm",
                    f"Similar to “{s['t']}” by {s['a'][0]} (shared by {s.get('by', '?')})", s["t"])
        for tag in top_genres[:2]:
            for i, (title, artist) in enumerate(lastfm_tag_tracks(tag)[:150]):
                add(title, artist, tag_rank_points(i + 1), "Last.fm", f"A standout in {tag}")
        for s in ([] if late else seeds[:3]):
            for title, artist, band, weight in listenbrainz_similar(s["a"][0]):
                add(title, artist, 0.6 * seed_weight(s, now) * weight, "ListenBrainz",
                    f"Listeners of {s['a'][0]} also play {artist}", s["t"])
        group_artists = {a for s in gsongs for a in s["a"]}
        picks, source = ([], None) if late else ai_picks(group, seeds, top_genres, group_artists or playlist_artist_names)
        if source:
            ai_used = source
        for title, artist, why in picks:
            add(title, artist, 1.2, "AI curator", why)
        for c in cands.values():                      # found by several sources = stronger
            c["score"] *= 1 + 0.4 * (len(c["src"]) - 1)

        # carry over this week's picks (stable within a week), drop ones posted since
        keep = []
        if same_week:
            for pg in prev.get("groups", []):
                if pg.get("name") == group:
                    keep = [p for p in pg.get("picks", []) if p.get("id") not in posted_ids
                            and p.get("id") not in in_playlist_ids]
        chosen, artists_used = list(keep), {norm(p["a"][0]) for p in keep}
        used_global |= {key(p["t"], p["a"][0]) for p in keep}
        used_ids |= {p["id"] for p in keep if p.get("id")}

        ranked = sorted(cands.values(), key=lambda c: -c["score"])
        checked = [0]

        def try_add(c):
            k = key(c["t"], c["a"])
            if k in used_global or k in shown_recently or norm(c["a"]) in artists_used:
                return False
            checked[0] += 1
            listeners = lastfm_listeners(c["t"], c["a"])
            if listeners is not None and listeners > MAX_LISTENERS:
                return False
            sp = spotify_find(c["t"], c["a"])
            if sp is None:                                 # no Spotify check available
                if listeners is None or listeners < 1:
                    return False                           # can't confirm it exists
                sp = {}
            elif not sp:
                return False                               # not on Spotify (or invented)
            tid = sp.get("id")
            if tid and (tid in in_playlist_ids or tid in posted_ids or tid in used_ids
                        or (tid in shown_recently_ids and not same_week)):
                return False
            niche = 1 / (1 + max(listeners or 0, 0) / NICHE_SCALE)
            why = next((w for s, w in c["why"] if s == "AI curator"), None) or (c["why"][0][1] if c["why"] else "")
            chosen.append({
                "t": sp.get("t") or c["t"], "a": sp.get("a") or [c["a"]], "id": sp.get("id", ""),
                "img": sp.get("img", ""), "al": sp.get("al", ""), "y": sp.get("y"),
                "why": why, "src": sorted(c["src"]), "lis": listeners if listeners and listeners > 0 else 0,
                "kind": "New artist" if norm(c["a"]) not in playlist_artists else "Deeper cut",
                "seeds": c["seeds"][:3], "score": round(c["score"] * niche, 3),
            })
            artists_used.add(norm(c["a"]))
            used_global.add(k)
            if tid:
                used_ids.add(tid)
            return True

        ai_have = sum(1 for p in chosen if "AI curator" in p.get("src", []))
        for c in ranked:                                   # 1) reserved slots for the AI curator
            if ai_have >= AI_SLOTS or len(chosen) >= PICKS_PER_GROUP or checked[0] >= 20:
                break
            if "AI curator" in c["src"] and try_add(c):
                ai_have += 1
        for c in ranked:                                   # 2) the rest by score
            if len(chosen) >= PICKS_PER_GROUP or checked[0] >= 60:
                break
            try_add(c)
        if len(keep) < len(chosen):                        # new picks: niche first, carried-over order kept
            chosen = keep + sorted(chosen[len(keep):], key=lambda p: -p["score"])
        log(f"  {len(cands)} candidates -> {len(chosen)} picks"
            + (f" ({len(keep)} kept from earlier this week)" if keep else ""))
        out_groups.append({"name": group, "slot": named.index(group) if group in named else -1,
                           "seeds": [{"t": s["t"], "a": s["a"][0], "by": s.get("by", "")} for s in seeds[:6]],
                           "genres": top_genres[:5], "picks": chosen[:PICKS_PER_GROUP], "quick": late})
        save(CACHE_FILE, CACHE, compact=True)          # keep progress even if the run is stopped
        log(f"  ({(time.time() - START) / 60:.1f} min elapsed)")

    # history + picked up
    for g in out_groups:
        for p in g["picks"]:
            h = history.setdefault(key(p["t"], p["a"][0]), {"t": p["t"], "a": p["a"][0], "id": p.get("id", ""),
                                                            "first": now.strftime("%Y-%m-%d"), "weeks": []})
            if week not in h["weeks"]:
                h["weeks"].append(week)
            h["group"] = g["name"]
    by_id = {s["id"]: s for s in songs}
    picked, picked_ids = [], set()
    for k, h in sorted(history.items(), key=lambda kv: kv[1]["first"]):
        s = by_id.get(h.get("id"))
        if s and s["id"] not in picked_ids and s.get("how") == "posted" and s.get("at", "")[:10] >= h["first"]:
            picked_ids.add(s["id"])
            picked.append({"t": s["t"], "a": s["a"], "id": s["id"], "img": s.get("img", ""), "by": s["by"],
                           "at": s["at"][:10], "suggested": h["first"], "group": h.get("group", "")})
    picked.sort(key=lambda p: p["at"], reverse=True)

    result = {"generated": now.strftime("%Y-%m-%dT%H:%M"), "week": week,
              "sources": {"lastfm": bool(LASTFM_KEY), "listenbrainz": True, "ai": ai_used,
                          "spotify_check": bool(_spotify["ok"])},
              "groups": out_groups, "picked_up": picked[:30]}
    save(ROOT / "suggestions.json", result, compact=True)
    save(ROOT / "suggest_history.json", history)
    save(CACHE_FILE, CACHE, compact=True)
    total = sum(len(g["picks"]) for g in out_groups)
    log(f"\nDone: {total} suggestions in {len(out_groups)} groups. AI: {ai_used or 'not used'}. "
        f"Picked up so far: {len(picked)}.")


if __name__ == "__main__":
    main()
