import { useCallback, useEffect, useRef, useState } from "react";
import { SearchBar } from "./SearchBar";
import { ProductCard } from "./ProductCard";
import { LoadingGrid, EmptyState, ErrorState } from "./SearchStates";
import { searchProducts, fetchSuggestions } from "../lib/api";
import { useDebounced } from "../lib/useDebounced";
import { Sparkles, Tag } from "lucide-react";

const EXAMPLES = [
  "nike",
  "samsung",
  "laptop",
  "nik shose",
  "samsng phone",
  "wireles hedphone",
  "something to carry my laptop",
  "device for listening to music",
  "shoes for morning running",
  "something to charge my phone",
  "bag for traveling",
];

// Simple bounded LRU-ish cache for autocomplete results
const suggestionCache = new Map();
const CACHE_MAX = 50;

function getCached(key) {
  return suggestionCache.get(key) ?? null;
}

function setCache(key, value) {
  if (suggestionCache.size >= CACHE_MAX) {
    // Delete oldest entry
    const firstKey = suggestionCache.keys().next().value;
    suggestionCache.delete(firstKey);
  }
  suggestionCache.set(key, value);
}

export function SearchPage() {
  // --- Typing state (drives autocomplete only) ---
  const [searchQuery, setSearchQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // --- Submitted state (drives product results) ---
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [results, setResults] = useState(null);
  const [interpretation, setInterpretation] = useState(null);
  const [searchStatus, setSearchStatus] = useState("idle"); // idle | loading | success | error
  const [searchError, setSearchError] = useState("");
  const [elapsed, setElapsed] = useState(0);

  // Debounced typing for autocomplete (fast, 200ms)
  const debouncedQuery = useDebounced(searchQuery, 200);

  // Abort controllers & request sequence tracker
  const suggestAbort = useRef(null);
  const searchAbort = useRef(null);
  const suggestSeqRef = useRef(0);

  // ==================================================================
  // Autocomplete: fires on debounced typing, NOT product search
  // ==================================================================
  useEffect(() => {
    const q = debouncedQuery.trim();
    if (!q || q.length < 2) {
      setSuggestions([]);
      setSuggestionsLoading(false);
      return;
    }

    // Check cache first
    const cached = getCached(q.toLowerCase());
    if (cached) {
      setSuggestions(cached);
      setSuggestionsLoading(false);
      return;
    }

    // Abort previous suggestion request and track sequence
    suggestAbort.current?.abort();
    const controller = new AbortController();
    suggestAbort.current = controller;
    const currentSeq = ++suggestSeqRef.current;

    setSuggestionsLoading(true);

    fetchSuggestions(q, { signal: controller.signal, limit: 8 })
      .then((data) => {
        if (currentSeq !== suggestSeqRef.current || controller.signal.aborted) return;
        const items = data?.suggestions ?? [];
        setSuggestions(items);
        setCache(q.toLowerCase(), items);
        setSuggestionsLoading(false);
      })
      .catch((err) => {
        if (currentSeq !== suggestSeqRef.current || controller.signal.aborted || err?.name === "AbortError") return;
        setSuggestions([]);
        setSuggestionsLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [debouncedQuery]);

  // ==================================================================
  // Product Search: fires ONLY on explicit submission
  // ==================================================================
  const runSearch = useCallback((query) => {
    const q = (query ?? "").trim();

    // Handle clear
    if (!q) {
      setSubmittedQuery("");
      setResults(null);
      setInterpretation(null);
      setSuggestions([]);
      setSearchStatus("idle");
      setSearchQuery("");
      return;
    }

    setSubmittedQuery(q);
    setSearchQuery(q);
    setSearchStatus("loading");
    setSearchError("");
    setShowSuggestions(false);

    // Abort previous search
    searchAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    const started = performance.now();

    searchProducts(q, { signal: controller.signal, limit: 12 })
      .then((data) => {
        if (controller.signal.aborted) return;
        setElapsed(performance.now() - started);
        const resList = data?.results ?? [];
        setResults(resList);
        setInterpretation(data?.interpretation ?? null);
        setSearchStatus("success");
      })
      .catch((err) => {
        if (controller.signal.aborted || err?.name === "AbortError") return;
        setSearchError(err?.message || "Unable to reach the search API.");
        setSearchStatus("error");
      });
  }, []);

  const hasInterpretation = interpretation && (
    interpretation.detected_brands?.length > 0 ||
    interpretation.detected_categories?.length > 0 ||
    interpretation.min_price != null ||
    interpretation.max_price != null ||
    interpretation.soft_preferences?.length > 0 ||
    Boolean(interpretation.did_you_mean)
  );

  return (
    <main className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="mx-auto max-w-6xl px-4 py-5 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-foreground">IntelliSearch</h1>
            <p className="text-sm text-muted-foreground">Offline AI Hybrid Product Search Engine</p>
          </div>
          <div className="hidden sm:flex items-center gap-1.5 rounded-full border border-border bg-muted/50 px-3 py-1 text-xs text-muted-foreground">
            <span className="font-medium text-foreground">ChromaDB + PostgreSQL</span>
            <span>· Exact · Partial · Fuzzy · Semantic</span>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-3xl px-4 pb-2 pt-10">
        <SearchBar
          value={searchQuery}
          onChange={setSearchQuery}
          onSubmit={runSearch}
          suggestions={suggestions}
          suggestionsLoading={suggestionsLoading}
          open={showSuggestions}
          setOpen={setShowSuggestions}
        />

        {/* Dynamic Query Interpretation Badge Strip */}
        {hasInterpretation && (
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-border/80 bg-muted/40 px-3 py-2 text-xs">
            <div className="flex items-center gap-1 font-medium text-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              <span>Query Understanding:</span>
            </div>
            {interpretation.did_you_mean && (
              <button
                type="button"
                onClick={() => {
                  setSearchQuery(interpretation.did_you_mean);
                  runSearch(interpretation.did_you_mean);
                }}
                className="group flex items-center gap-1 rounded bg-amber-500/10 px-2 py-0.5 font-medium text-amber-700 dark:text-amber-300 hover:bg-amber-500/20 transition-colors"
                title="Click to search corrected query"
              >
                <span>Did you mean:</span>
                <span className="underline italic group-hover:text-amber-800 dark:group-hover:text-amber-200">
                  {interpretation.did_you_mean}
                </span>
              </button>
            )}
            {interpretation.detected_brands?.map((b) => (
              <span key={b} className="rounded bg-primary/10 px-2 py-0.5 font-medium text-primary">
                Brand: {b}
              </span>
            ))}
            {interpretation.detected_categories?.map((c) => (
              <span key={c} className="rounded bg-secondary px-2 py-0.5 font-medium text-secondary-foreground">
                Category: {c}
              </span>
            ))}
            {(interpretation.min_price != null || interpretation.max_price != null) && (
              <span className="rounded bg-emerald-500/10 px-2 py-0.5 font-medium text-emerald-600 dark:text-emerald-400">
                Price:{" "}
                {interpretation.min_price != null && interpretation.max_price != null
                  ? `₹${Number(interpretation.min_price).toLocaleString("en-IN")} – ₹${Number(interpretation.max_price).toLocaleString("en-IN")}`
                  : interpretation.min_price != null
                  ? `≥ ₹${Number(interpretation.min_price).toLocaleString("en-IN")}`
                  : `≤ ₹${Number(interpretation.max_price).toLocaleString("en-IN")}`}
              </span>
            )}
            {interpretation.soft_preferences?.map((pref) => (
              <span key={pref} className="rounded bg-muted px-2 py-0.5 text-muted-foreground">
                Preference: {pref}
              </span>
            ))}
          </div>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">Try:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              suppressHydrationWarning
              onClick={() => runSearch(ex)}
              className="rounded-full border border-border bg-card px-3 py-1 text-xs text-foreground hover:border-primary hover:text-primary transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 pb-20 pt-8">
        {searchStatus === "loading" && <LoadingGrid />}

        {searchStatus === "error" && (
          <ErrorState message={searchError} onRetry={() => runSearch(submittedQuery)} />
        )}

        {searchStatus === "success" && (
          <>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-sm text-muted-foreground">
              <div className="flex items-center gap-x-4">
                <span className="font-medium text-foreground">"{submittedQuery}"</span>
                <span>{results.length} results</span>
                <span>{elapsed.toFixed(0)} ms</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Hybrid Vector Retrieval &amp; Reranking
              </div>
            </div>
            {results.length === 0 ? (
              <EmptyState query={submittedQuery} />
            ) : (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {results.map((p) => (
                  <ProductCard key={p.id} product={p} />
                ))}
              </div>
            )}
          </>
        )}

        {searchStatus === "idle" && (
          <p className="py-16 text-center text-sm text-muted-foreground">
            Start typing to see suggestions, then press Enter to search 7,500 products.
          </p>
        )}
      </section>
    </main>
  );
}
