"""
ChromaDB Vector & Embedding Viewer — NorthStar Product Search

Section: 📦 Browse Product Vectors
"""

import sys
from pathlib import Path

# Ensure backend root is available in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import streamlit as st

from app.services.embedding_sync import (
    MODEL_NAME,
    get_chroma_collection,
)


@st.cache_resource
def load_collection():
    """Cache the persistent ChromaDB collection."""
    try:
        collection = get_chroma_collection(create_if_missing=False)
        return collection, None
    except Exception as e:
        return None, str(e)


def run_app():
    st.set_page_config(
        page_title="ChromaDB Vector Viewer — NorthStar",
        page_icon="📦",
        layout="wide",
    )

    collection, error = load_collection()

    if error or collection is None:
        st.error(f"❌ Could not load ChromaDB collection: {error}")
        st.info(
            "💡 **To initialize ChromaDB**, run:\n"
            "```bash\n"
            "python backend/scripts/generate_embeddings.py\n"
            "```\n"
            "or inside Docker:\n"
            "```bash\n"
            "docker compose exec -T backend python scripts/sync_embeddings.py\n"
            "```"
        )
        return

    # ---------------------------------------------------------
    # 📦 Browse Product Vectors
    # ---------------------------------------------------------
    st.subheader("📦 Browse Product Vectors")

    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        filter_text = st.text_input(
            "Filter by product name, brand, or tag keyword:",
            placeholder="e.g. Nike, Logitech, Laptop...",
        )
    with b_col2:
        b_limit = st.slider("Display Limit:", min_value=5, max_value=50, value=10, step=5)

    if filter_text.strip():
        browse_data = collection.get(
            where_document={"$contains": filter_text.strip()},
            limit=b_limit,
            include=["embeddings", "documents", "metadatas"],
        )
    else:
        browse_data = collection.get(
            limit=b_limit,
            include=["embeddings", "documents", "metadatas"],
        )

    b_ids = browse_data.get("ids", [])
    b_docs = browse_data.get("documents", [])
    b_metas = browse_data.get("metadatas", [])
    b_vecs = browse_data.get("embeddings", [])

    if not b_ids:
        st.warning(f"No vectors found matching '{filter_text}'.")
    else:
        st.success(f"Displaying **{len(b_ids)}** stored product vectors:")
        for i, p_id in enumerate(b_ids):
            meta = b_metas[i] if (b_metas is not None and len(b_metas) > i and b_metas[i] is not None) else {}
            doc = b_docs[i] if (b_docs is not None and len(b_docs) > i and b_docs[i] is not None) else ""
            vec = b_vecs[i] if (b_vecs is not None and len(b_vecs) > i and b_vecs[i] is not None) else []
            vec_arr = np.array(vec, dtype=np.float32)
            vec_norm = float(np.linalg.norm(vec_arr)) if len(vec_arr) > 0 else 0.0

            p_name = meta.get("product_name", f"Product #{p_id}")
            p_brand = meta.get("brand", "—")
            p_cat = meta.get("category", "—")
            p_price = meta.get("price", 0.0)
            p_hash = meta.get("embedding_hash", "—")

            with st.expander(f"📌 **{p_name}** | Brand: `{p_brand}` | Category: `{p_cat}` | ₹{p_price:,.2f} (ID: {p_id})"):
                col_l, col_r = st.columns([1, 1])
                with col_l:
                    st.markdown("#### 📋 Stored Metadata & Hash")
                    st.json({
                        "product_id": p_id,
                        "product_name": p_name,
                        "brand": p_brand,
                        "category": p_cat,
                        "price": p_price,
                        "embedding_hash": p_hash,
                        "embedding_model": meta.get("embedding_model", MODEL_NAME),
                    })
                    st.markdown("**Searchable Document Text:**")
                    st.info(doc if doc else "No text stored.")

                with col_r:
                    st.markdown("#### 🧬 Embedding Vector (384-dim)")
                    st.write(f"- **Dimensions:** `{len(vec)}`")
                    st.write(f"- **L2 Norm:** `{vec_norm:.6f}` (Unit normalized: `{abs(vec_norm - 1.0) < 1e-4}`)")
                    if len(vec) >= 6:
                        st.write(f"- **First 6 floats:** `[{', '.join(f'{x:.4f}' for x in vec[:6])}, ...]`")
                    st.text_area(
                        f"Vector Array #{p_id}",
                        value=str(vec),
                        height=120,
                        key=f"b_vec_{p_id}_{i}",
                    )


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if ctx is not None:
        run_app()
    else:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())