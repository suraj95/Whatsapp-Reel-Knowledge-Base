import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="WhatsApp Reel Knowledge Base", layout="centered")

st.title("WhatsApp Reel Knowledge Base")

tabs = st.tabs(["Save new reel", "Ask questions"])

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
                resp = requests.post(
                    f"{API_BASE}/reels",
                    json={"url": url, "manual_tags": tags_list},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success("Reel saved successfully!")
                st.write("**Summary:**")
                st.write(data["summary"])
                st.write("**Tags:** " + ", ".join(data["auto_tags"]))
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
                        st.markdown(f"**Summary:** {r['summary']}")
                        st.markdown(f"**Tags:** {', '.join(r['auto_tags'])}")
                        st.caption(f"Reel ID: {r['reel_id']}")
            except Exception as e:
                st.error(f"Error querying reels: {e}")

