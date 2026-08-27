import sys
from pathlib import Path

# Base Directory Configurations
BACKEND_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BACKEND_DIR / "data" / "chroma"
COLLECTION_NAME = "products"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chromadb
import numpy as np
import streamlit as st


@st.cache_resource
def get_chroma_collection():
    """Cache the persistent ChromaDB client and collection."""
    if not CHROMA_DIR.exists():
        return None, f"ChromaDB directory not found at `{CHROMA_DIR}`."
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        return collection, None
    except Exception as e:
        return None, str(e)


def run_app():
    st.set_page_config(
        page_title="ChromaDB Vector Viewer — NorthStar Search",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 ChromaDB Vector & Embedding Viewer")
    st.caption("Offline Intelligent Product Search — Local Vector Index Inspection")

    collection, error = get_chroma_collection()

    if error or collection is None:
        st.error(f"❌ Could not load ChromaDB collection: {error}")
        st.info(
            "💡 **To generate embeddings**, run:\n"
            "```bash\n"
            "python backend/scripts/generate_embeddings.py\n"
            "```\n"
            "or inside Docker:\n"
            "```bash\n"
            "docker compose exec backend python scripts/generate_embeddings.py\n"
            "```"
        )
        return

    # ---------------------------------------------------------
    # Statistics Summary Header
    # ---------------------------------------------------------
    count = collection.count()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Collection Name", COLLECTION_NAME)
    with col2:
        st.metric("Total Embeddings", f"{count:,}")
    with col3:
        st.metric("Vector Dimension", "384 (float32)")
    with col4:
        st.metric("Storage Engine", "ChromaDB Persistent (HNSW)")

    st.divider()

    # ---------------------------------------------------------
    # Tabs for Inspection and Live Vector Testing
    # ---------------------------------------------------------
    tab_inspect, tab_similarity = st.tabs(["📦 Browse Product Embeddings", "🎯 Live Vector Cosine Similarity"])

    with tab_inspect:
        st.subheader("Browse Stored Product Vectors")
        search_col, filter_col = st.columns([3, 1])

        with search_col:
            search_text = st.text_input(
                "Search by title, brand, or tag keyword:",
                placeholder="e.g. Nike, Samsung, Headphones, Backpack...",
            )
        with filter_col:
            limit = st.slider("Result Limit:", min_value=5, max_value=50, value=15, step=5)

        if search_text.strip():
            results = collection.get(
                where_document={"$contains": search_text.strip()},
                limit=limit,
                include=["embeddings", "documents", "metadatas"],
            )
        else:
            results = collection.get(
                limit=limit,
                include=["embeddings", "documents", "metadatas"],
            )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings", [])

        if not ids:
            st.warning(f"No products found matching '{search_text}'.")
        else:
            st.success(f"Displaying **{len(ids)}** product vectors:")

            for i, product_id in enumerate(ids):
                metadata = metadatas[i] or {}
                document = documents[i] or ""
                vector = embeddings[i] if embeddings is not None and len(embeddings) > i else []
                vec_arr = np.array(vector, dtype=np.float32)
                vec_norm = float(np.linalg.norm(vec_arr)) if len(vec_arr) > 0 else 0.0

                product_name = metadata.get("product_name", f"Product #{product_id}")
                brand = metadata.get("brand", "—")
                category = metadata.get("category", "—")
                price = metadata.get("price", 0.0)

                with st.expander(f"📌 **{product_name}** | Brand: `{brand}` | Category: `{category}` | Price: ₹{price:,.2f} (ID: {product_id})"):
                    left, right = st.columns([1, 1])

                    with left:
                        st.markdown("#### 📋 Metadata & Document Text")
                        st.json({
                            "id": product_id,
                            "product_name": product_name,
                            "brand": brand,
                            "category": category,
                            "price": price,
                        })
                        st.markdown("**Constructed Searchable Document:**")
                        st.info(document if document else "No text document attached.")

                    with right:
                        st.markdown("#### 🧬 Embedding Vector (384-dim)")
                        st.write(f"- **Dimensions:** `{len(vector)}`")
                        st.write(f"- **L2 Norm:** `{vec_norm:.6f}` (Unit normalized: `{abs(vec_norm - 1.0) < 1e-4}`)")
                        if len(vector) >= 6:
                            st.write(f"- **First 6 floats:** `[{', '.join(f'{x:.4f}' for x in vector[:6])}, ...]`")
                        
                        st.markdown("**Full Vector Float Array:**")
                        st.text_area(
                            f"Vector #{product_id}",
                            value=str(vector),
                            height=120,
                            key=f"vec_{product_id}_{i}",
                        )

    with tab_similarity:
        st.subheader("🎯 Test Query Cosine Similarity Live")
        st.caption("Embed an ad-hoc query string and calculate real-time Cosine Similarity against all vectors in ChromaDB.")

        test_query = st.text_input(
            "Enter a search query to test semantic vector retrieval:",
            value="wireless noise canceling headphones",
        )
        sim_limit = st.slider("Top-K Candidates:", min_value=3, max_value=20, value=5)

        if st.button("Calculate Vector Similarities", type="primary"):
            try:
                from fastembed import TextEmbedding
                model = TextEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_dir=str(BACKEND_DIR / "models" / "all-MiniLM-L6-v2"),
                )
                query_vec = list(model.embed([test_query]))[0]
                q_arr = np.array(query_vec, dtype=np.float32)
                q_norm = np.linalg.norm(q_arr)
                if q_norm > 0:
                    q_arr = q_arr / q_norm

                sim_res = collection.query(
                    query_embeddings=[q_arr.tolist()],
                    n_results=sim_limit,
                    include=["metadatas", "documents", "distances"],
                )

                if sim_res and sim_res.get("ids") and sim_res["ids"][0]:
                    ret_ids = sim_res["ids"][0]
                    ret_dists = sim_res["distances"][0]
                    ret_metas = sim_res["metadatas"][0]

                    st.markdown(f"### Top Matches for *\"{test_query}\"*")
                    for rank, (p_id, dist, meta) in enumerate(zip(ret_ids, ret_dists, ret_metas), 1):
                        cosine_sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                        st.markdown(
                            f"**#{rank}. {meta.get('product_name')}** (ID: `{p_id}`) — **Cosine Similarity:** `{cosine_sim:.4f}` (L2² Distance: `{dist:.4f}`)\n"
                            f"- Brand: `{meta.get('brand')}` | Category: `{meta.get('category')}` | Price: `₹{meta.get('price', 0):,.2f}`"
                        )
            except Exception as e:
                st.error(f"Error executing vector inference: {e}")


if __name__ == "__main__":
    # Check if executed via `streamlit run` or directly via `python`
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if ctx is not None:
        # Running inside Streamlit server
        run_app()
    else:
        # User executed via standard `python backend/scripts/chroma_viewer.py`
        # Automatically bootstrap Streamlit CLI!
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())