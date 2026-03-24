import requests
import streamlit as st
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="WhatsApp Reel Knowledge Base", layout="centered")

st.title("WhatsApp Reel Knowledge Base")

tabs = st.tabs(["Save new reel", "Ask questions"])


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
        st.markdown(f"**Summary:** {summary}")
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

