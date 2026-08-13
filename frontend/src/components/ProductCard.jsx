import { useState } from "react";
import { ChevronDown, ImageOff } from "lucide-react";
import { BASE_URL } from "../lib/api";

function fmtScore(n) {
  return typeof n === "number" ? n.toFixed(2) : "—";
}

function imageUrl(src) {
  if (!src) return null;
  return /^https?:\/\//.test(src) ? src : `${BASE_URL}${src}`;
}

const SCORE_ROWS = [
  ["Final score", "score"],
  ["Exact score", "exact_score"],
  ["Prefix score", "prefix_score"],
  ["Fuzzy score", "fuzzy_score"],
  ["Semantic score", "semantic_score"],
];

export function ProductCard({ product }) {
  const [expanded, setExpanded] = useState(false);
  const [broken, setBroken] = useState(false);
  const src = imageUrl(product.image);

  return (
    <article className="flex flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div className="flex aspect-[4/3] items-center justify-center bg-muted">
        {src && !broken ? (
          <img
            src={src}
            alt={product.name}
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
          <h3 className="text-sm font-semibold leading-snug text-foreground">{product.name}</h3>
          <span className="shrink-0 rounded-md bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
            {fmtScore(product.score)}
          </span>
        </div>

        <p className="text-xs text-muted-foreground">
          {product.brand}
          {product.category ? ` · ${product.category}` : ""}
        </p>

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
            Search Score
            <ChevronDown className={`h-4 w-4 ${expanded ? "rotate-180" : ""}`} />
          </button>
          {expanded && (
            <dl className="mt-2 space-y-1">
              {SCORE_ROWS.map(([label, key]) => (
                <div key={key} className="flex justify-between text-xs">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="font-mono text-foreground">{fmtScore(product[key])}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      </div>
    </article>
  );
}
