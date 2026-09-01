"""
ChromaDB Vector & Embedding Viewer — NorthStar Product Search

Single Source of Truth:
------------------------
This Streamlit tool is a diagnostic and inspection interface over the EXACT
backend search primitives. It reuses backend modules directly:
- Embedding Model: FastEmbed 'sentence-transformers/all-MiniLM-L6-v2' (384-dim)
- Text Construction: app.services.embedding_sync.build_product_text
- Query Embedding: app.services.semantic_search._get_query_embedding
- Semantic Search: app.services.semantic_search.semantic_search_products
- Hybrid Ranking: app.services.search_ranking.search_products
- Query Parsing: app.services.query_parser.parse_query
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure backend root is available in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import chromadb
import numpy as np
import streamlit as st
from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.product import Product
from app.services.embedding_sync import (
    COLLECTION_NAME,
    CHROMA_DIR,
    MODEL_NAME,
    EMBEDDING_DIMENSION,
    build_product_text,
    calculate_product_hash,
    get_chroma_collection,
    synchronize_embeddings,
)
from app.services.query_parser import parse_query
from app.services.search_ranking import search_products
from app.services.semantic_search import (
    get_semantic_search_resources,
    _get_query_embedding,
    semantic_search_products,
)


@st.cache_resource
def load_backend_resources():
    """Cache the persistent ChromaDB collection and FastEmbed model."""
    try:
        model, collection = get_semantic_search_resources()
        return model, collection, None
    except Exception as e:
        return None, None, str(e)


def run_app():
    st.set_page_config(
        page_title="ChromaDB Vector & Embedding Viewer — NorthStar",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔍 ChromaDB Vector & Embedding Inspector")
    st.caption("NorthStar Product Search — Real-Time Vector Index & Semantic Diagnostic Interface")

    model, collection, error = load_backend_resources()

    if error or collection is None:
        st.error(f"❌ Could not load ChromaDB backend resources: {error}")
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
    # Live Dynamic Metric Header
    # ---------------------------------------------------------
    db = SessionLocal()
    try:
        pg_count = db.query(Product).count()
    except Exception:
        pg_count = 0
    finally:
        db.close()

    chroma_count = collection.count()
    
    # Dynamically measure vector dimension from a stored vector
    dim_str = f"{EMBEDDING_DIMENSION} (float32)"
    try:
        sample = collection.get(limit=1, include=["embeddings"])
        sample_vecs = sample.get("embeddings") if sample else None
        if sample_vecs is not None and len(sample_vecs) > 0:
            actual_dim = len(sample_vecs[0])
            dim_str = f"{actual_dim} (float32)"
    except Exception:
        actual_dim = EMBEDDING_DIMENSION

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Collection Name", COLLECTION_NAME)
    with col2:
        st.metric("ChromaDB Vectors", f"{chroma_count:,}")
    with col3:
        st.metric("PostgreSQL Products", f"{pg_count:,}")
    with col4:
        st.metric("Vector Dimension", dim_str)
    with col5:
        in_sync = (chroma_count == pg_count and chroma_count > 0)
        st.metric("Catalog Status", "✅ In Sync" if in_sync else "⚠️ Out of Sync")

    st.divider()

    # ---------------------------------------------------------
    # Sidebar: Backend Configuration & Diagnostics
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Backend Configuration")
        st.markdown(f"**Embedding Model:** `{MODEL_NAME}`")
        st.markdown(f"**Vector Dimensions:** `{EMBEDDING_DIMENSION}`")
        st.markdown(f"**Storage Directory:** `{CHROMA_DIR.name}/`")
        st.markdown(f"**Collection:** `{COLLECTION_NAME}`")
        st.markdown(f"**Distance Metric:** `Squared L2 (||u - v||²)`")
        st.markdown(f"**Normalization:** `L2 Unit (||v|| = 1.0)`")
        st.markdown(f"**Cosine Conversion:** `1.0 - (dist / 2.0)`")

        st.divider()
        st.subheader("🔍 Quick Sync Trigger")
        if st.button("Run Incremental Sync", type="secondary"):
            with st.spinner("Synchronizing PostgreSQL catalog with ChromaDB..."):
                sync_db = SessionLocal()
                try:
                    res = synchronize_embeddings(sync_db)
                    st.success(f"Sync complete: +{res.new_count} new, ~{res.updated_count} updated, -{res.deleted_count} deleted.")
                    st.rerun()
                except Exception as sync_err:
                    st.error(f"Sync error: {sync_err}")
                finally:
                    sync_db.close()

    # ---------------------------------------------------------
    # Main Tabs
    # ---------------------------------------------------------
    tab_sim, tab_browse, tab_consistency, tab_comparison = st.tabs([
        "🎯 Vector Similarity & Search",
        "📦 Browse Product Vectors",
        "⚖️ Backend Consistency Check",
        "🔬 Search Pipeline Inspector",
    ])

    # ---------------------------------------------------------
    # TAB 1: Vector Similarity & Search
    # ---------------------------------------------------------
    with tab_sim:
        st.subheader("🎯 Test Query Semantic Similarity & Vector Retrieval")
        st.caption("Directly inspects the exact backend FastEmbed vector generation and ChromaDB retrieval.")

        query_col, limit_col = st.columns([3, 1])
        with query_col:
            test_query = st.text_input(
                "Enter search query:",
                value="nyykes shoos",
                help="Test typo queries, exact names, or natural language descriptions.",
            )
        with limit_col:
            top_k = st.slider("Top-K Candidates:", min_value=3, max_value=25, value=5)

        search_mode = st.radio(
            "Evaluation Mode:",
            [
                "1. Raw Vector Similarity (Direct ChromaDB query on raw text)",
                "2. Backend Semantic Search (With query parsing & typo-variant merging)",
                "3. Full Hybrid Search (Semantic + Exact + Fuzzy + Partial + Brands)",
            ],
            index=0,
            horizontal=True,
        )

        if st.button("Execute Vector Search", type="primary"):
            eval_db = SessionLocal()
            try:
                t0 = time.perf_counter()

                if search_mode.startswith("1."):
                    # Raw Vector Search (direct query embedding)
                    q_vec = _get_query_embedding(test_query.strip())
                    raw_res = collection.query(
                        query_embeddings=[q_vec.tolist()],
                        n_results=top_k,
                        include=["metadatas", "documents", "distances"],
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    st.markdown(f"#### Results: Raw Vector Query for *\"{test_query}\"* ({elapsed_ms:.1f}ms)")
                    st.info("💡 **Note:** Raw vector search feeds the uncorrected string directly into the embedding model. Typo tokens (e.g. `nyykes`) produce low out-of-vocabulary similarities.")

                    if raw_res and raw_res.get("ids") and raw_res["ids"][0]:
                        r_ids = raw_res["ids"][0]
                        r_dists = raw_res["distances"][0]
                        r_metas = raw_res["metadatas"][0]
                        r_docs = raw_res["documents"][0]

                        for rank, (pid, dist, meta, doc) in enumerate(zip(r_ids, r_dists, r_metas, r_docs), 1):
                            cosine_sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                            pname = meta.get("product_name", f"Product #{pid}")
                            brand = meta.get("brand", "—")
                            cat = meta.get("category", "—")
                            price = meta.get("price", 0.0)

                            st.markdown(
                                f"**#{rank}. {pname}** (ID: `{pid}`) — **Cosine Similarity:** `{cosine_sim:.4f}` | L2² Distance: `{dist:.4f}`\n"
                                f"- Brand: `{brand}` | Category: `{cat}` | Price: `₹{price:,.2f}`"
                            )
                            with st.expander(f"Inspect Document Text (ID: {pid})"):
                                st.code(doc)

                elif search_mode.startswith("2."):
                    # Backend Semantic Search with Query Parser & Variant Merging
                    parsed = parse_query(eval_db, test_query)
                    sem_results = semantic_search_products(
                        eval_db,
                        test_query,
                        limit=top_k,
                        normalized_query=parsed.semantic_query,
                        normalized_queries=[parsed.semantic_query, parsed.normalized_query],
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    st.markdown(f"#### Results: Backend Semantic Search for *\"{test_query}\"* ({elapsed_ms:.1f}ms)")
                    st.markdown(f"- **Normalized Semantic Query:** `{parsed.semantic_query}` | **Detected Brands:** `{parsed.detected_brands}`")

                    if not sem_results:
                        st.warning("No semantic search results found.")
                    else:
                        for rank, sr in enumerate(sem_results, 1):
                            p = sr.product
                            st.markdown(
                                f"**#{rank}. {p.product_name}** (ID: `{p.id}`) — **Semantic Similarity:** `{sr.semantic_score:.4f}`\n"
                                f"- Brand: `{p.brand}` | Category: `{p.category}` | Price: `₹{float(p.price):,.2f}`"
                            )

                else:
                    # Full Hybrid Search
                    ranked_results, parsed = search_products(eval_db, test_query, limit=top_k)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    st.markdown(f"#### Results: Full Backend Hybrid Search for *\"{test_query}\"* ({elapsed_ms:.1f}ms)")
                    st.markdown(f"- **Parsed Semantic Query:** `{parsed.semantic_query}` | **Did you mean:** `{parsed.did_you_mean}`")

                    if not ranked_results:
                        st.warning("No hybrid search results found.")
                    else:
                        for rank, hr in enumerate(ranked_results, 1):
                            p = hr.product
                            st.markdown(
                                f"**#{rank}. {p.product_name}** (ID: `{p.id}`) — **Final Score:** `{hr.final_score:.4f}`\n"
                                f"- **Breakdown:** Semantic: `{hr.semantic_score:.4f}` | Fuzzy: `{hr.fuzzy_score:.4f}` | Exact: `{hr.exact_score:.4f}` | Partial: `{hr.partial_score:.4f}` | Brand Match: `{hr.brand_match}`"
                            )

            finally:
                eval_db.close()

    # ---------------------------------------------------------
    # TAB 2: Browse Product Vectors
    # ---------------------------------------------------------
    with tab_browse:
        st.subheader("📦 Browse Stored Product Vectors & Metadata")
        b_col1, b_col2 = st.columns([3, 1])
        with b_col1:
            filter_text = st.text_input("Filter by product name, brand, or tag keyword:", placeholder="e.g. Nike, Logitech, Laptop...")
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
                meta = b_metas[i] or {}
                doc = b_docs[i] or ""
                vec = b_vecs[i] if (b_vecs is not None and len(b_vecs) > i) else []
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

    # ---------------------------------------------------------
    # TAB 3: Backend Consistency Check
    # ---------------------------------------------------------
    with tab_consistency:
        st.subheader("⚖️ Backend ↔ Streamlit Architecture Consistency Diagnostic")
        st.caption("Verifies that Streamlit uses the exact same model, configuration, and data structures as the backend.")

        diag_db = SessionLocal()
        try:
            # 1. Model Verification
            model_ok = (MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2")
            
            # 2. Collection Verification
            coll_ok = (collection.name == COLLECTION_NAME)
            
            # 3. Dimension Verification
            sample_res = collection.get(limit=1, include=["embeddings"])
            s_embeddings = sample_res.get("embeddings") if sample_res else None
            dim_val = len(s_embeddings[0]) if (s_embeddings is not None and len(s_embeddings) > 0) else 0
            dim_ok = (dim_val == EMBEDDING_DIMENSION)

            # 4. Catalog Count Verification
            db_count = diag_db.query(Product).count()
            count_ok = (chroma_count == db_count)

            # 5. Unit Normalization Verification
            q_test = _get_query_embedding("test vector normalization")
            q_norm = float(np.linalg.norm(q_test))
            q_norm_ok = abs(q_norm - 1.0) < 1e-4

            # 6. Product Text Builder Verification
            sample_prod = diag_db.query(Product).first()
            if sample_prod:
                txt = build_product_text(sample_prod)
                h = calculate_product_hash(txt)
                builder_ok = bool(txt and h and len(h) == 64)
            else:
                builder_ok = True

            data = [
                {"Component": "Embedding Model", "Backend Value": MODEL_NAME, "Streamlit Value": MODEL_NAME, "Status": "✅ MATCH" if model_ok else "❌ MISMATCH"},
                {"Component": "Embedding Dimension", "Backend Value": str(EMBEDDING_DIMENSION), "Streamlit Value": str(dim_val), "Status": "✅ MATCH" if dim_ok else "❌ MISMATCH"},
                {"Component": "ChromaDB Collection", "Backend Value": COLLECTION_NAME, "Streamlit Value": collection.name, "Status": "✅ MATCH" if coll_ok else "❌ MISMATCH"},
                {"Component": "Vector Count vs DB", "Backend Value": f"{db_count} products", "Streamlit Value": f"{chroma_count} vectors", "Status": "✅ IN SYNC" if count_ok else "⚠️ SYNC NEEDED"},
                {"Component": "Query L2 Normalization", "Backend Value": "||v|| = 1.0", "Streamlit Value": f"||v|| = {q_norm:.6f}", "Status": "✅ UNIT NORM" if q_norm_ok else "❌ NOT NORMALIZED"},
                {"Component": "Product Text Builder", "Backend Value": "app.services.embedding_sync", "Streamlit Value": "Reused canonical function", "Status": "✅ SHARED" if builder_ok else "❌ SEPARATE"},
            ]
            st.table(data)

            if all([model_ok, coll_ok, dim_ok, q_norm_ok, builder_ok]):
                st.success("🎉 All core embedding and vector search primitives are 100% consistent with the backend!")
            else:
                st.warning("⚠️ One or more configuration items differ.")

        finally:
            diag_db.close()

    # ---------------------------------------------------------
    # TAB 4: Search Pipeline Comparison
    # ---------------------------------------------------------
    with tab_comparison:
        st.subheader("🔬 Benchmark Query Pipeline Comparison")
        st.caption("Compare Raw ChromaDB Vector Retrieval vs Full Backend Hybrid Search across standard benchmark queries.")

        benchmark_queries = [
            "nyykes shoos",
            "wireless mouse",
            "logitech mouse",
            "gaming laptop",
            "something to eat or drink",
            "asdkjhaskjdh",
        ]

        bench_db = SessionLocal()
        try:
            for bq in benchmark_queries:
                with st.expander(f"🔎 Query: *\"{bq}\"*"):
                    # A. Raw vector search
                    q_v = _get_query_embedding(bq)
                    raw_ret = collection.query(query_embeddings=[q_v.tolist()], n_results=3, include=["metadatas", "distances"])
                    
                    # B. Backend hybrid search
                    hybrid_ret, parsed_q = search_products(bench_db, bq, limit=3)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**1. Raw ChromaDB Vector Search (Unprocessed Query):**")
                        if raw_ret and raw_ret.get("ids") and raw_ret["ids"][0]:
                            for rk, (pid, dst, m) in enumerate(zip(raw_ret["ids"][0], raw_ret["distances"][0], raw_ret["metadatas"][0]), 1):
                                sim = max(0.0, min(1.0, 1.0 - (float(dst) / 2.0)))
                                st.write(f"#{rk}. **{m.get('product_name')}** (ID: {pid}) — `sim={sim:.4f}`, `dist={dst:.4f}`")
                        else:
                            st.write("No vector results.")

                    with col_b:
                        st.markdown(f"**2. Backend Search Pipeline (`query='{parsed_q.semantic_query}'`):**")
                        if hybrid_ret:
                            for rk, hr in enumerate(hybrid_ret, 1):
                                st.write(f"#{rk}. **{hr.product.product_name}** (ID: {hr.product.id}) — `score={hr.final_score:.4f}` (`sem={hr.semantic_score:.4f}`, `fuzzy={hr.fuzzy_score:.4f}`)")
                        else:
                            st.write("0 results returned (correctly gated).")

        finally:
            bench_db.close()


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