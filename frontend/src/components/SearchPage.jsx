import { useCallback, useEffect, useRef, useState } from "react";
import { SearchBar } from "./SearchBar";
import { ProductCard } from "./ProductCard";
import { LoadingGrid, EmptyState, ErrorState } from "./SearchStates";
import { searchProducts } from "../lib/api";
import { useDebounced } from "../lib/useDebounced";

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
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);

  const debounced = useDebounced(input, 300);
  const searchAbort = useRef(null);

  const runSearch = useCallback((query) => {
    const q = (query ?? "").trim();
    if (!q) {
      setSubmitted("");
      setResults(null);
      setSuggestions([]);
      setStatus("idle");
      return;
    }

    setSubmitted(q);
    setInput(q);
    setStatus("loading");
    setError("");

    searchAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    const started = performance.now();

    searchProducts(q, { signal: controller.signal, limit: 10 })
      .then((data) => {
        if (controller.signal.aborted) return;
        setElapsed(performance.now() - started);
        const resList = data?.results ?? [];
        setResults(resList);
        setSuggestions(resList);
        setStatus("success");
      })
      .catch((err) => {
        if (controller.signal.aborted || err?.name === "AbortError") return;
        setError(err?.message || "Unable to reach the search API.");
        setStatus("error");
      });
  }, []);

  // Debounced search effect with AbortController for stale response prevention
  useEffect(() => {
    const q = debounced.trim();
    if (!q) {
      setSuggestions([]);
      setResults(null);
      setStatus("idle");
      return;
    }

    searchAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    const started = performance.now();

    setStatus("loading");
    setSubmitted(q);
    setError("");

    searchProducts(q, { signal: controller.signal, limit: 10 })
      .then((data) => {
        if (controller.signal.aborted) return;
        setElapsed(performance.now() - started);
        const resList = data?.results ?? [];
        setResults(resList);
        setSuggestions(resList);
        setStatus("success");
      })
      .catch((err) => {
        if (controller.signal.aborted || err?.name === "AbortError") return;
        setError(err?.message || "Unable to reach the search API.");
        setStatus("error");
      });

    return () => {
      controller.abort();
    };
  }, [debounced]);

  return (
    <main className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="mx-auto max-w-6xl px-4 py-5 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-foreground">IntelliSearch</h1>
            <p className="text-sm text-muted-foreground">Offline Intelligent Product Search</p>
          </div>
          <div className="hidden sm:flex items-center gap-1.5 rounded-full border border-border bg-muted/50 px-3 py-1 text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Hybrid Engine</span>
            <span>· Exact · Partial · Fuzzy · Semantic</span>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-3xl px-4 pb-2 pt-10">
        <SearchBar
          value={input}
          onChange={setInput}
          onSubmit={runSearch}
          suggestions={suggestions}
          loading={status === "loading"}
          open={open}
          setOpen={setOpen}
        />

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
              <div className="sm:hidden text-xs text-muted-foreground">
                Hybrid Search (Exact · Partial · Fuzzy · Semantic)
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
            Start typing to search the product index using the hybrid ranking engine.
          </p>
        )}
      </section>
    </main>
  );
}
