import requests
import streamlit as st
import pandas as pd
import base64
from html import escape
from pathlib import Path
import re

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="WhatsApp Reel Knowledge Base", layout="centered")

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
          }

          .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: 1px solid rgba(30,58,95,0.14) !important;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.20), rgba(251, 191, 36, 0.18)) !important;
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
          .travel-summary-list {
            margin: 0;
            padding-left: 18px;
          }
          .travel-summary-list li {
            margin: 6px 0;
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

    # Travel-friendly bot logo from open-source icons (embedded as data URIs).
    bot_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
      stroke="#1e3a5f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 8V4H8" />
      <rect width="16" height="12" x="4" y="8" rx="2" />
      <path d="M2 14h2" />
      <path d="M20 14h2" />
      <path d="M15 13v2" />
      <path d="M9 13v2" />
    </svg>
    """.strip()
    compass_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
      stroke="#1e3a5f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="m16.24 7.76-1.804 5.411a2 2 0 0 1-1.265 1.265L7.76 16.24l1.804-5.411a2 2 0 0 1 1.265-1.265z" />
    </svg>
    """.strip()

    bot_data_uri = "data:image/svg+xml;base64," + base64.b64encode(bot_svg.encode("utf-8")).decode("utf-8")
    compass_data_uri = (
        "data:image/svg+xml;base64,"
        + base64.b64encode(compass_svg.encode("utf-8")).decode("utf-8")
    )

    st.markdown(
        f"""
        <div class="travel-header">
          <div aria-hidden="true" style="display:flex; align-items:center;">
            <div class="travel-robot">
              <img
                class="robot-base"
                alt=""
                src="{bot_data_uri}"
              />
              <img
                class="robot-badge"
                alt=""
                src="{compass_data_uri}"
              />
            </div>
          </div>
          <div>
            <div class="travel-title">WhatsApp Reel Knowledge Base</div>
            <div class="travel-subtitle">Travel notes powered by AI summaries</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_inject_travel_theme()

tabs = st.tabs(["Save new reel", "Ask questions"])


def _summary_to_numbered_items(summary: str, max_items: int = 6) -> list[str]:
    """
    Convert a raw LLM/enrichment summary into a clean list of items.

    - If the text already contains newlines/bullets, use those as items.
    - Otherwise split into sentences and keep the first few.
    """
    if not summary:
        return []

    s = str(summary).strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    def _clean_item(t: str) -> str:
        t = t.strip()
        # Remove leading bullets or numbering like "1.", "2)", "- ", "• "
        t = re.sub(r"^(?:\d+[\.\)]|[-*•])\s*", "", t).strip()
        return t

    # Case 1: multi-line content -> use each non-empty line as an item
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if len(lines) > 1:
        items = [_clean_item(ln) for ln in lines if _clean_item(ln)]
        return items[:max_items]

    # Case 2: single line -> split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", s)
    sentences = [x.strip() for x in sentences if x.strip()]
    if len(sentences) > 1:
        return sentences[:max_items]

    # Case 3: fallback split by common separators
    parts = [p.strip() for p in re.split(r"\s*(?:;|/|•|-)\s*", s) if p.strip()]
    return parts[:max_items] if parts else [s]


def render_enrichment(enrichment: dict):
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
        items = _summary_to_numbered_items(summary)
        if items:
            items_html = "".join(f"<li>{escape(item)}</li>" for item in items)
            st.markdown(
                f"""
                <div class="travel-summary-boundary">
                  <div class="travel-summary-title">Summary</div>
                  <ol class="travel-summary-list">
                    {items_html}
                  </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if tags:
        st.markdown("**tags:** " + ", ".join(tags))

    if lat is not None and lng is not None:
        st.caption("Approximate location")
        try:
            map_df = pd.DataFrame([{"lat": float(lat), "lon": float(lng)}])
            st.map(map_df, zoom=11)
        except Exception:
            st.caption(f"Map coordinates: {lat}, {lng}")

# ---------- TAB 1: SAVE REEL ----------
with tabs[0]:
    st.subheader("Save a new reel")
    url = st.text_input("Paste reel link")
    manual_tags = st.text_input(
        "Optional manual tags (comma separated)",
        placeholder="goa, restaurant, bali, street food",
    )

    if st.button("Save reel", type="primary"):
        if not url:
            st.error("Please paste a reel URL.")
        else:
            tags_list = [t.strip() for t in manual_tags.split(",") if t.strip()] if manual_tags else []
            try:
                with st.spinner("Analyzing reel, enriching details, and saving..."):
                    resp = requests.post(
                        f"{API_BASE}/reels",
                        json={"url": url, "manual_tags": tags_list},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                st.success("Reel saved successfully!")
                if data.get("enrichment"):
                    render_enrichment(data["enrichment"])
                st.caption(f"Reel ID: {data['reel_id']}")
            except Exception as e:
                st.error(f"Error saving reel: {e}")

# ---------- TAB 2: ASK QUESTIONS ----------
with tabs[1]:
    st.subheader("Ask your saved reels")
    query = st.text_input(
        "Ask something...",
        placeholder="show restaurants we saved in Goa",
    )
    top_k = st.slider("How many results?", 1, 10, 5)

    if st.button("Search reels"):
        if not query:
            st.error("Please enter a question.")
        else:
            try:
                with st.spinner("Searching your reel knowledge base..."):
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={"query": query, "top_k": top_k},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                results = data.get("results", [])

                if not results:
                    st.info("No matching reels found yet. Try saving some first.")
                else:
                    for r in results:
                        st.markdown("---")
                        st.markdown(f"**Score:** `{r['score']:.3f}`")
                        st.markdown(f"**URL:** {r['url']}")
                        if r.get("enrichment"):
                            render_enrichment(r["enrichment"])
                        st.caption(f"Reel ID: {r['reel_id']}")
            except Exception as e:
                st.error(f"Error querying reels: {e}")

