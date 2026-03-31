import requests
import streamlit as st
import pandas as pd
import base64
import json
from pathlib import Path
import re
import time
import pydeck as pdk
from constants import CHAT_TOP_K

API_BASE = "http://localhost:8000"
FAVICON_PATH = Path(__file__).resolve().parents[1] / "docs" / "images" / "flight_4283062.png"

st.set_page_config(
    page_title="Travel Reels Knowledge Base",
    page_icon=str(FAVICON_PATH),
    layout="centered",
)

def _inject_travel_theme() -> None:
    # Use a local image asset as the background texture.
    # Keeping it as a small-ish PNG avoids huge base64 strings.
    repo_root = Path(__file__).resolve().parents[1]
    # Use a smaller background texture to keep the injected CSS size manageable.
    bg_path = repo_root / "docs" / "images" / "International-travel.jpeg"
    bg_b64 = ""
    try:
        bg_b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
    except Exception:
        bg_b64 = ""

    bg_img_css = (
        f'url("data:image/png;base64,{bg_b64}")' if bg_b64 else "none"
    )

    css = """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;600&display=swap');

          .travel-bg {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background:
              radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.22), transparent 45%),
              radial-gradient(circle at 80% 10%, rgba(251, 191, 36, 0.18), transparent 40%),
              radial-gradient(circle at 70% 85%, rgba(34, 197, 94, 0.14), transparent 45%),
              linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.65));
          }

          .travel-bg::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image: {bg_img_css};
            background-size: cover;
            background-position: center;
            opacity: 0.35;
            filter: saturate(1.08) blur(0.6px);
            transform: scale(1.02);
            pointer-events: none;
          }

          .travel-bg::after {
            content: "";
            position: absolute;
            inset: 0;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'%3E%3Cg fill='none' stroke='%231e3a5f' stroke-opacity='0.12' stroke-width='1'%3E%3Cpath d='M30 110 C 60 80, 90 80, 120 110 S 180 140, 210 110'/%3E%3Cpath d='M10 40 C 40 10, 70 10, 100 40 S 160 70, 190 40'/%3E%3Cpath d='M10 180 C 40 150, 70 150, 100 180 S 160 210, 190 180'/%3E%3Ccircle cx='55' cy='150' r='3'/%3E%3Ccircle cx='165' cy='70' r='3'/%3E%3Ccircle cx='120' cy='120' r='2'/%3E%3C/g%3E%3C/svg%3E");
            background-size: 220px 220px;
            opacity: 1;
            pointer-events: none;
          }

          html, body {
            background: transparent !important;
          }

          [data-testid="stAppViewContainer"] {
            font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            color: rgba(15, 23, 42, 0.95) !important;
            position: relative;
            z-index: 1;
            background: transparent !important;
          }

          .block-container {
            color: rgba(15, 23, 42, 0.95) !important;
            background: transparent !important;
          }

          h1, h2, h3, h4, p, span {
            color: rgba(15, 23, 42, 0.95) !important;
          }

          .travel-header {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 14px 18px;
            margin-top: 8px;
            margin-bottom: 18px;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(30,58,95,0.10);
            box-shadow: 0 10px 30px rgba(15,23,42,0.06);
          }

          .travel-brand {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(251, 191, 36, 0.18));
            border: 1px solid rgba(30,58,95,0.12);
          }

          .travel-brand .travel-airplane {
            position: absolute;
            right: -6px;
            bottom: -8px;
            opacity: 0.28;
          }

          .travel-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-weight: 700;
            font-size: 1.65rem;
            line-height: 1.15;
            margin: 0;
          }

          .travel-subtitle {
            color: rgba(30,58,95,0.72);
            font-weight: 400;
            margin-top: 2px;
            font-size: 0.95rem;
          }

          div[data-testid="stTextInput"] input,
          div[data-baseweb="select"] select,
          textarea {
            border-radius: 12px !important;
            border: 2px solid rgba(30,58,95,0.38) !important;
            background: rgba(255,255,255,0.92) !important;
            box-shadow: 0 2px 10px rgba(15,23,42,0.05);
          }
          div[data-testid="stTextInput"] input:focus,
          textarea:focus {
            border: 2px solid rgba(14,165,233,0.85) !important;
            box-shadow: 0 0 0 2px rgba(14,165,233,0.18) !important;
          }

          .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: 2px solid rgba(30,58,95,0.28) !important;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.20), rgba(251, 191, 36, 0.18)) !important;
          }

          /* Messages / "chat-like" feedback boxes */
          div[data-testid="stAlert"] {
            border: 2px solid rgba(30,58,95,0.28) !important;
            border-radius: 14px !important;
            background: rgba(255,255,255,0.88) !important;
          }

          /* Chat bubbles */
          div[data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.78) !important;
            border: 1px solid rgba(30,58,95,0.12) !important;
            border-radius: 14px !important;
            box-shadow: 0 6px 18px rgba(15,23,42,0.05);
          }

          /* Make map area clearly bounded */
          div[data-testid="stMap"],
          div[data-testid="stDeckGlJsonChart"],
          div[data-testid="stVegaLiteChart"] {
            border: 2px solid rgba(30,58,95,0.34) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 18px rgba(15,23,42,0.08);
          }

          /* Numbered result boxes */
          .travel-result-title {
            font-weight: 800;
            color: rgba(30,58,95,0.95);
            margin-bottom: 8px;
          }

          /* Keep chat composer fixed to viewport bottom */
          div[data-testid="stChatInput"] {
            position: fixed !important;
            left: 50%;
            transform: translateX(-50%);
            bottom: max(10px, env(safe-area-inset-bottom));
            width: min(736px, calc(100vw - 2rem));
            z-index: 9999;
            padding: 0.35rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.78);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(30,58,95,0.16);
            box-shadow: 0 10px 30px rgba(15,23,42,0.14);
          }
          .block-container {
            padding-bottom: 9rem !important;
          }

          /* Travel summary boundary (numbered list) */
          .travel-summary-boundary {
            border-radius: 16px;
            border: 1px solid rgba(30,58,95,0.12);
            background: rgba(255,255,255,0.68);
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            margin-top: 8px;
          }
          .travel-summary-title {
            font-weight: 700;
            color: rgba(30,58,95,0.92);
            margin-bottom: 6px;
          }
          .travel-summary-text {
            margin: 0;
            line-height: 1.5;
            color: rgba(15,23,42,0.92);
          }

          /* Fancy robotic animation */
          .travel-robot {
            width: 64px;
            height: 64px;
            animation: robot-wiggle 3.4s ease-in-out infinite;
            transform-origin: 50% 60%;
            position: relative;
            display: grid;
            place-items: center;
          }
          @keyframes robot-wiggle {
            0%, 100% { transform: rotate(-2deg) translateY(0); }
            50% { transform: rotate(2deg) translateY(-2px); }
          }
          .travel-robot img {
            display: block;
          }
          .travel-robot img.robot-base {
            width: 64px;
            height: 64px;
            opacity: 0.92;
            filter: drop-shadow(0 10px 18px rgba(15,23,42,0.10));
          }
          .travel-robot img.robot-badge {
            width: 22px;
            height: 22px;
            position: absolute;
            right: 6px;
            top: 6px;
            opacity: 0.75;
          }
        </style>
        <div class="travel-bg"></div>
        """
    css = css.replace("{bg_img_css}", bg_img_css)
    st.markdown(css, unsafe_allow_html=True)

    # Homepage icon (from Flaticon link shared by user), embedded as data URI.
    # Fall back to a simple emoji if the local asset is unavailable.
    icon_path = repo_root / "docs" / "images" / "flight_4283062.png"
    flight_icon_data_uri = ""
    try:
        flight_icon_data_uri = (
            "data:image/png;base64,"
            + base64.b64encode(icon_path.read_bytes()).decode("utf-8")
        )
    except Exception:
        flight_icon_data_uri = ""
    icon_html = (
        f'<img class="robot-base" alt="Flight icon" src="{flight_icon_data_uri}" />'
        if flight_icon_data_uri
        else '<div style="font-size:34px; line-height:1;">✈️</div>'
    )

    st.markdown(
        f"""
        <div class="travel-header">
          <div aria-hidden="true" style="display:flex; align-items:center;">
            <div class="travel-robot">
              {icon_html}
            </div>
          </div>
          <div>
            <div class="travel-title">Travel Reels Knowledge Base</div>
            <div class="travel-subtitle">Travel notes powered by AI summaries</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_inject_travel_theme()

tabs = st.tabs(["Save new reel", "Ask questions"])

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Ask me about your saved reels and I will find matches.\n\n"
                "Try: `show restaurants we saved in Goa`"
            ),
            "results": None,
            "agentic": None,
        }
    ]
if "map_already_shown" not in st.session_state:
    st.session_state.map_already_shown = False


def narrative_json_to_markdown(text: str) -> str:
    """If the model returned a JSON object as narrative, render as readable Markdown."""
    raw = (text or "").strip()
    if not raw:
        return ""
    candidate = raw
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return raw
    try:
        data = json.loads(candidate)
    except Exception:
        return raw
    if not isinstance(data, dict):
        return raw
    parts = []
    for k, v in data.items():
        title = str(k).replace("_", " ").strip().title()
        if isinstance(v, dict):
            parts.append(f"### {title}")
            for sk, sv in v.items():
                parts.append(f"- **{sk}:** {sv}")
        elif isinstance(v, list):
            parts.append(f"### {title}")
            for item in v:
                parts.append(f"- {item}")
        else:
            parts.append(f"### {title}\n\n{v}")
    return "\n\n".join(parts)


def _narrative_looks_like_json_object(text: str) -> bool:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return len(t) > 2 and t.startswith("{") and t.endswith("}")


def render_enrichment(enrichment: dict, show_map: bool = True):
    place_name = enrichment.get("place_name") or "Unknown place"
    city = enrichment.get("city")
    country = enrichment.get("country")
    lat = enrichment.get("lat")
    lng = enrichment.get("lng")
    rating = enrichment.get("rating")
    category = enrichment.get("category")
    cuisine = enrichment.get("cuisine")
    price_range = enrichment.get("price_range")
    summary = enrichment.get("summary")
    tags = enrichment.get("tags", [])

    location_parts = [part for part in [city, country] if part]
    if location_parts:
        st.markdown(f"**Place:** {place_name} ({', '.join(location_parts)})")
    else:
        st.markdown(f"**Place:** {place_name}")

    info_cols = st.columns(3)
    info_cols[0].markdown(f"**Category:** {category or 'n/a'}")
    info_cols[1].markdown(f"**Price:** {price_range or 'n/a'}")
    info_cols[2].markdown(f"**Rating:** {rating if rating is not None else 'n/a'}")

    if cuisine:
        st.markdown(f"**Cuisine:** {cuisine}")
    if summary:
        summary_text = str(summary).strip()
        summary_text = re.sub(r"^(?:\d+[\.\)]|[-*•])\s*", "", summary_text)
        summary_text = summary_text.replace("\n", "<br/>")
        st.markdown(
            f"""
            <div class="travel-summary-boundary">
              <div class="travel-summary-title">Summary</div>
              <p class="travel-summary-text">{summary_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if tags:
        st.markdown("**tags:** " + ", ".join(tags))

    if show_map and lat is not None and lng is not None:
        st.caption("Approximate location")
        try:
            map_df = pd.DataFrame([{"lat": float(lat), "lon": float(lng)}])
            st.map(map_df, zoom=11)
        except Exception:
            st.caption(f"Map coordinates: {lat}, {lng}")


def render_map_points(map_points: list, *, intent: str = "search"):
    if not map_points:
        return
    # Second response is often trip_planning with more pins; fingerprint matching fails.
    # Skip map entirely on trip follow-ups once any map has been shown this session.
    if intent == "trip_planning" and st.session_state.get("map_already_shown"):
        return
    rows = []
    for p in map_points:
        lat, lng = p.get("lat"), p.get("lng")
        if lat is None or lng is None:
            continue
        try:
            lat_f = float(lat)
            lng_f = float(lng)
        except Exception:
            continue
        label = p.get("city") or p.get("place_name") or "Place"
        rows.append({"lat": lat_f, "lng": lng_f, "label": label})
    if not rows:
        return

    df = pd.DataFrame(rows)
    center_lat = float(df["lat"].mean())
    center_lng = float(df["lng"].mean())
    zoom = 10 if len(rows) == 1 else 7

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lng, lat]",
        get_radius=12,
        radius_units="pixels",
        stroked=True,
        get_line_width=2,
        get_line_color=[255, 255, 255, 240],
        get_fill_color=[220, 38, 38, 230],
        pickable=True,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=df,
        get_position="[lng, lat]",
        get_text="label",
        get_size=18,
        get_color=[255, 240, 80, 255],
        get_alignment_baseline="'top'",
        get_pixel_offset=[0, 14],
    )
    deck = pdk.Deck(
        layers=[scatter_layer, text_layer],
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=zoom, pitch=0),
        tooltip={"text": "{label}"},
    )
    cap = "Locations on your results" if len(rows) > 1 else "Top location"
    st.caption(cap)
    st.pydeck_chart(deck, use_container_width=True)
    st.session_state.map_already_shown = True


def render_intent_cards(intent: str, cards: list):
    if not cards:
        return
    if intent == "trip_planning":
        st.markdown("### Itinerary")
    elif intent == "recommendation":
        st.markdown("### Recommendations")
    else:
        st.markdown("### Results")

    for card in cards:
        title = card.get("title") or "Untitled"
        subtitle = card.get("subtitle")
        summary = card.get("summary")
        reason = card.get("reason")
        st.markdown(f"**{title}**")
        if subtitle:
            st.caption(subtitle)
        if summary:
            st.markdown(summary)
        if reason:
            st.caption(reason)

        metadata = card.get("metadata") or {}
        places = metadata.get("places") or []
        if places:
            for idx, place in enumerate(places, start=1):
                place_name = place.get("place_name") or "Unknown place"
                location = ", ".join([x for x in [place.get("city"), place.get("country")] if x])
                line = f"{idx}. {place_name}"
                if location:
                    line += f" ({location})"
                st.markdown(line)
        forecast = metadata.get("forecast")
        if forecast:
            st.caption("Forecast (daily)")
            if isinstance(forecast, list) and forecast:
                lines = ["| Date | High °C | Low °C | Rain % |", "| --- | --- | --- | --- |"]
                for row in forecast[:8]:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        "| {date} | {hi} | {lo} | {rain} |".format(
                            date=row.get("date", "—"),
                            hi=row.get("max_c", "—"),
                            lo=row.get("min_c", "—"),
                            rain=row.get("precip_pct", "—"),
                        )
                    )
                st.markdown("\n".join(lines))
            else:
                st.json(forecast)
        offers = metadata.get("offers")
        if offers:
            st.caption("Sample flight offers")
            if isinstance(offers, list):
                for i, off in enumerate(offers[:5], start=1):
                    if not isinstance(off, dict):
                        continue
                    price = off.get("price_total")
                    cur = off.get("currency", "")
                    segs = off.get("segments") or []
                    seg_bits = []
                    for s in segs[:2]:
                        if isinstance(s, dict):
                            seg_bits.append(
                                f"{s.get('from', '?')} → {s.get('to', '?')} ({s.get('flight', '')})"
                            )
                    st.markdown(
                        f"{i}. **{price} {cur}**  \n   {', '.join(seg_bits) if seg_bits else 'See offer details'}"
                    )
            else:
                st.json(offers)
        stops = metadata.get("stops")
        if stops:
            for s in stops[:8]:
                nm = s.get("name") or "Stop"
                k = s.get("kind") or ""
                st.markdown(f"- {nm} ({k})")
        pois = metadata.get("pois")
        if pois:
            for poi in pois[:8]:
                nm = poi.get("name") or "Place"
                k = poi.get("kind") or ""
                st.markdown(f"- {nm} ({k})")
        st.divider()


def split_narrative_and_question(text: str):
    text = (text or "").strip()
    if not text:
        return "", ""
    question_match = re.search(r"([^?]*\?)\s*$", text)
    if not question_match:
        return text, ""
    question = question_match.group(1).strip()
    body = text[: question_match.start()].strip()
    return body, question


def render_results_boxes(results: list):
    if not results:
        return
    st.markdown("#### Results")
    for idx, r in enumerate(results, start=1):
        enrichment = r.get("enrichment") or {}
        place_name = enrichment.get("place_name") or "Saved reel"
        city = enrichment.get("city")
        country = enrichment.get("country")
        location = ", ".join([x for x in [city, country] if x])
        title = f"Result #{idx} - {place_name}"
        if location:
            title += f" ({location})"

        with st.expander(title, expanded=False):
            summary = enrichment.get("summary") or r.get("summary") or "No summary available."
            score = r.get("score", 0.0)
            st.markdown(
                f"<div style='font-size:0.86rem; color:#334155;'>"
                f"Score: {float(score):.3f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:0.88rem; line-height:1.45;'>{summary}</div>",
                unsafe_allow_html=True,
            )
            url = r.get("url")
            if url:
                st.markdown(
                    f"<div style='font-size:0.86rem; margin-top:6px;'>"
                    f"<a href='{url}' target='_blank'>Open reel</a></div>",
                    unsafe_allow_html=True,
                )

# ---------- TAB 1: SAVE REEL ----------
with tabs[0]:
    st.subheader("Save a new reel")
    url = st.text_input("Paste reel link")

    if st.button("Save reel", type="primary"):
        if not url:
            st.error("Please paste a reel URL.")
        else:
            try:
                with st.spinner("Analyzing reel, enriching details, and saving..."):
                    resp = requests.post(
                        f"{API_BASE}/reels",
                        json={"url": url},
                        timeout=20,
                    )
                    resp.raise_for_status()
                    create_data = resp.json()
                    job_id = create_data["job_id"]

                    data = None
                    start = time.time()
                    while time.time() - start < 180:
                        status_resp = requests.get(
                            f"{API_BASE}/reels/{job_id}/status",
                            timeout=15,
                        )
                        status_resp.raise_for_status()
                        status_data = status_resp.json()
                        if status_data.get("status") == "completed":
                            data = status_data.get("result")
                            break
                        if status_data.get("status") == "failed":
                            raise RuntimeError(status_data.get("error") or "Reel ingestion failed.")
                        time.sleep(2)
                    if not data:
                        raise RuntimeError("Reel ingestion is still running. Please retry shortly.")
                st.success("Reel saved successfully!")
                if data.get("enrichment"):
                    render_enrichment(data["enrichment"])
                st.caption(f"Reel ID: {data['reel_id']}")
            except Exception as e:
                st.error(f"Error saving reel: {e}")

# ---------- TAB 2: ASK QUESTIONS ----------
with tabs[1]:
    st.subheader("Chat with your saved reels")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
                continue

            content = message.get("content") or ""
            body_text, question_text = split_narrative_and_question(content)
            base = body_text or content
            if _narrative_looks_like_json_object(base):
                st.markdown(narrative_json_to_markdown(base))
            else:
                st.markdown(base)
            if question_text:
                st.markdown(question_text)

            if message.get("agentic"):
                agentic = message["agentic"]
                intent = agentic.get("intent", "search")
                cards = agentic.get("cards") or []
                if cards:
                    render_intent_cards(intent, cards)
                if intent in ("search", "recommendation"):
                    st.session_state.map_already_shown = False
                render_map_points(agentic.get("map_points", []), intent=intent)
            if message.get("results"):
                render_results_boxes(message["results"])

    prompt = st.chat_input("Ask something about your reels...")
    if prompt:
        conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt, "results": None, "agentic": None}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                with st.spinner("Thinking..."):
                    resp = requests.post(
                        f"{API_BASE}/query-agentic",
                        json={
                            "query": prompt,
                            "top_k": CHAT_TOP_K,
                            "conversation_history": conversation_history,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                intent = data.get("intent", "search")
                cards = data.get("cards", [])
                map_points = data.get("map_points", [])
                results = data.get("sources", [])
                narrative = data.get("narrative") or ""

                if not results:
                    if intent == "trip_planning":
                        full_reply = narrative or (
                            "I couldn't pull saved reels for this destination yet. "
                            "Save a few reels or name a place more clearly, and try again."
                        )
                    else:
                        full_reply = (
                            "I could not find matching reels yet. Save a few reels first, then try again."
                        )
                else:
                    full_reply = narrative

                def typewriter_stream(text: str):
                    for token in text.split(" "):
                        yield token + " "
                        time.sleep(0.02)

                if _narrative_looks_like_json_object(full_reply):
                    placeholder.markdown(narrative_json_to_markdown(full_reply))
                else:
                    body_text, question_text = split_narrative_and_question(full_reply)
                    placeholder.write_stream(typewriter_stream(body_text or full_reply))
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": full_reply,
                        "results": results,
                        "agentic": {
                            "intent": intent,
                            "cards": cards,
                            "map_points": map_points,
                        },
                    }
                )

                if cards:
                    render_intent_cards(intent, cards)
                if intent in ("search", "recommendation"):
                    st.session_state.map_already_shown = False
                if map_points:
                    render_map_points(map_points, intent=intent)

                if results:
                    render_results_boxes(results)
            except Exception as e:
                error_text = f"I hit an error while searching your reels: {e}"
                placeholder.markdown(error_text)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": error_text, "results": None, "agentic": None}
                )

