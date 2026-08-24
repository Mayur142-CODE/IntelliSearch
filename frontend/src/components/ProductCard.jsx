import { useState } from "react";
import { ChevronDown, ImageOff, Tag } from "lucide-react";
import { BASE_URL } from "../lib/api";

function fmtScore(n) {
  return typeof n === "number" ? n.toFixed(2) : "—";
}

function imageUrl(src) {
  if (!src) return null;
  return /^https?:\/\//.test(src) ? src : `${BASE_URL}${src}`;
}

const SCORE_ROWS = [
  ["Final score", "final_score"],
  ["Exact score", "exact_score"],
  ["Partial score", "partial_score"],
  ["Fuzzy score", "fuzzy_score"],
  ["Semantic score", "semantic_score"],
  ["Preference boost", "preference_score"],
];

export function ProductCard({ product }) {
  const [expanded, setExpanded] = useState(false);
  const [broken, setBroken] = useState(false);
  const src = imageUrl(product.image);

  const productName = product.product_name || product.name;
  const displayScore = product.final_score ?? product.fuzzy_score ?? product.score;

  const tagsList = product.tags
    ? (Array.isArray(product.tags) ? product.tags : product.tags.split(",")).slice(0, 3)
    : [];

  return (
    <article className="flex flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm transition-all hover:shadow-md">
      <div className="flex aspect-[4/3] items-center justify-center bg-muted">
        {src && !broken ? (
          <img
            src={src}
            alt={productName}
            loading="lazy"
            onError={() => setBroken(true)}
            className="h-full w-full object-cover"
          />
        ) : (
          <ImageOff className="h-8 w-8 text-muted-foreground" aria-hidden="true" />
        )}
      </div>

      <div className="flex flex-1 flex-col gap-2 p-4">
        <div className="flex items-start justify-between gap-2">
          <h3 className="text-sm font-semibold leading-snug text-foreground line-clamp-2">{productName}</h3>
          <span className="shrink-0 rounded-md bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
            {fmtScore(displayScore)}
          </span>
        </div>

        <p className="text-xs text-muted-foreground">
          <span className="font-medium text-foreground/80">{product.brand}</span>
          {product.category ? ` · ${product.category}` : ""}
        </p>

        {product.description && (
          <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
            {product.description}
          </p>
        )}

        {tagsList.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-1">
            {tagsList.map((t, idx) => (
              <span key={idx} className="rounded bg-muted/60 px-1.5 py-0.5 text-[10px] text-muted-foreground">
                #{t.trim()}
              </span>
            ))}
          </div>
        )}

        <p className="mt-auto pt-2 text-base font-semibold text-foreground">
          ₹{Number(product.price ?? 0).toLocaleString("en-IN")}
        </p>

        <div className="border-t border-border pt-2">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            className="flex w-full items-center justify-between text-xs font-medium text-primary"
          >
            Relevance Breakdown
            <ChevronDown className={`h-4 w-4 ${expanded ? "rotate-180" : ""}`} />
          </button>
          {expanded && (
            <dl className="mt-2 space-y-1">
              {SCORE_ROWS.map(([label, key]) => {
                const val = product[key];
                if (val === undefined || val === null) return null;
                return (
                  <div key={key} className="flex justify-between text-xs">
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="font-mono text-foreground">{fmtScore(val)}</dd>
                  </div>
                );
              })}
              {product.candidate_sources && product.candidate_sources.length > 0 && (
                <div className="flex justify-between text-xs pt-1 border-t border-border/50">
                  <dt className="text-muted-foreground">Sources</dt>
                  <dd className="font-mono text-xs text-primary">{product.candidate_sources.join(", ")}</dd>
                </div>
              )}
            </dl>
          )}
        </div>
      </div>
    </article>
  );
}
