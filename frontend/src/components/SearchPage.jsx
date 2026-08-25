import { useCallback, useEffect, useRef, useState } from "react";
import { SearchBar } from "./SearchBar";
import { ProductCard } from "./ProductCard";
import { LoadingGrid, EmptyState, ErrorState } from "./SearchStates";
import { searchProducts } from "../lib/api";
import { useDebounced } from "../lib/useDebounced";
import { SlidersHorizontal, Sparkles, Tag } from "lucide-react";

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

export function SearchPage() {
  const [input, setInput] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [open, setOpen] = useState(false);

  const [suggestions, setSuggestions] = useState([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [interpretation, setInterpretation] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);

  const debounced = useDebounced(input, 250);
  const searchAbort = useRef(null);
  const autocompleteAbort = useRef(null);

  // Committed search: called ONLY when user presses Enter, clicks a suggestion, or clicks an Example chip
  const runSearch = useCallback((query) => {
    const q = (query ?? "").trim();
    if (!q) {
      setSubmitted("");
      setResults(null);
      setInterpretation(null);
      setSuggestions([]);
      setStatus("idle");
      setOpen(false);
      return;
    }

    setSubmitted(q);
    setInput(q);
    setStatus("loading");
    setError("");
    setOpen(false); // Close autocomplete dropdown upon search submission

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
        setStatus("success");
      })
      .catch((err) => {
        if (controller.signal.aborted || err?.name === "AbortError") return;
        setError(err?.message || "Unable to reach the search API.");
        setStatus("error");
      });
  }, []);

  const handleClear = useCallback(() => {
    setInput("");
    setSubmitted("");
    setSuggestions([]);
    setResults(null);
    setInterpretation(null);
    setOpen(false);
    setStatus("idle");
  }, []);

  // Autocomplete suggestion fetch: triggers while typing to populate dropdown, WITHOUT rendering product results
  useEffect(() => {
    const q = debounced.trim();
    if (!q) {
      setSuggestions([]);
      setSuggestionsLoading(false);
      return;
    }

    autocompleteAbort.current?.abort();
    const controller = new AbortController();
    autocompleteAbort.current = controller;
    setSuggestionsLoading(true);

    searchProducts(q, { signal: controller.signal, limit: 8 })
      .then((data) => {
        if (controller.signal.aborted) return;
        const resList = data?.results ?? [];
        setSuggestions(resList);
        setSuggestionsLoading(false);
      })
      .catch((err) => {
        if (controller.signal.aborted || err?.name === "AbortError") return;
        setSuggestions([]);
        setSuggestionsLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [debounced]);

  const hasInterpretation = interpretation && (
    interpretation.detected_brands?.length > 0 ||
    interpretation.detected_categories?.length > 0 ||
    interpretation.min_price != null ||
    interpretation.max_price != null ||
    interpretation.soft_preferences?.length > 0
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
          value={input}
          onChange={setInput}
          onSubmit={runSearch}
          onClear={handleClear}
          suggestions={suggestions}
          loading={status === "loading" || suggestionsLoading}
          open={open}
          setOpen={setOpen}
        />

        {/* Dynamic Query Interpretation Badge Strip */}
        {hasInterpretation && (
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-border/80 bg-muted/40 px-3 py-2 text-xs">
            <div className="flex items-center gap-1 font-medium text-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              <span>Query Understanding:</span>
            </div>
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
                Price: {interpretation.min_price != null ? `≥ ₹${interpretation.min_price}` : ""} {interpretation.max_price != null ? `≤ ₹${interpretation.max_price}` : ""}
              </span>
            )}
            {interpretation.soft_preferences?.map((pref) => (
              <span key={pref} className="rounded bg-muted px-2 py-0.5 text-muted-foreground">
                Intent: {pref}
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
        {status === "loading" && <LoadingGrid />}

        {status === "error" && (
          <ErrorState message={error} onRetry={() => runSearch(submitted)} />
        )}

        {status === "success" && (
          <>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-sm text-muted-foreground">
              <div className="flex items-center gap-x-4">
                <span className="font-medium text-foreground">“{submitted}”</span>
                <span>{results.length} results</span>
                <span>{elapsed.toFixed(0)} ms</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Hybrid Vector Retrieval & Reranking
              </div>
            </div>
            {results.length === 0 ? (
              <EmptyState query={submitted} />
            ) : (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {results.map((p) => (
                  <ProductCard key={p.id} product={p} />
                ))}
              </div>
            )}
          </>
        )}

        {status === "idle" && (
          <p className="py-16 text-center text-sm text-muted-foreground">
            Start typing to search 7,500 products using the dynamic ChromaDB hybrid engine.
          </p>
        )}
      </section>
    </main>
  );
}
