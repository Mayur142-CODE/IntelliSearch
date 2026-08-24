import sys
from pathlib import Path

import chromadb
import streamlit as st

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BACKEND_DIR / "data" / "chroma"

COLLECTION_NAME = "products"

# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    COLLECTION_NAME
)

# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

st.set_page_config(
    page_title="ChromaDB Vector Viewer",
    layout="wide"
)

st.title("ChromaDB Vector Viewer")

st.caption(
    "Offline Product Search — Local Embedding Inspection"
)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

count = collection.count()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Collection",
        COLLECTION_NAME
    )

with col2:
    st.metric(
        "Total Vectors",
        count
    )

with col3:
    st.metric(
        "Embedding Dimension",
        "384"
    )

st.divider()

# ---------------------------------------------------------
# Search products
# ---------------------------------------------------------

st.subheader("Product Embeddings")

search_text = st.text_input(
    "Search product name / ID",
    placeholder="Example: Nike"
)

if search_text:

    results = collection.get(
        where_document={
            "$contains": search_text
        },
        limit=20,
        include=[
            "embeddings",
            "documents",
            "metadatas"
        ]
    )

else:

    results = collection.get(
        limit=20,
        include=[
            "embeddings",
            "documents",
            "metadatas"
        ]
    )

ids = results.get("ids", [])
documents = results.get("documents", [])
metadatas = results.get("metadatas", [])
embeddings = results.get("embeddings", [])

if not ids:

    st.warning("No products found.")

else:

    st.write(
        f"Showing {len(ids)} products"
    )

    for index, product_id in enumerate(ids):

        metadata = metadatas[index]
        document = documents[index]
        vector = embeddings[index]

        product_name = metadata.get(
            "product_name",
            "Unknown Product"
        )

        with st.expander(
            f"{product_name}  |  ID: {product_id}"
        ):

            left, right = st.columns(2)

            with left:

                st.markdown("### Product")

                st.write(
                    f"**ID:** {product_id}"
                )

                st.write(
                    f"**Name:** {metadata.get('product_name', '')}"
                )

                st.write(
                    f"**Brand:** {metadata.get('brand', '')}"
                )

                st.write(
                    f"**Category:** {metadata.get('category', '')}"
                )

                st.write(
                    f"**Price:** {metadata.get('price', '')}"
                )

                st.markdown("### Document")

                st.code(
                    document,
                    language="text"
                )

            with right:

                st.markdown(
                    "### Embedding Vector"
                )

                st.write(
                    f"Dimensions: **{len(vector)}**"
                )

                st.code(
                    str(vector),
                    language="text"
                )