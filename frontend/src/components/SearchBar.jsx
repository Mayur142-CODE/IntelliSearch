import { useEffect, useRef, useState } from "react";
import { Search, X, Loader2 } from "lucide-react";

export function SearchBar({
  value,
  onChange,
  onSubmit,
  suggestions = [],
  loading = false,
  open,
  setOpen,
}) {
  const [activeIndex, setActiveIndex] = useState(-1);
  const wrapperRef = useRef(null);

  useEffect(() => setActiveIndex(-1), [suggestions]);

  useEffect(() => {
    function onClickOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [setOpen]);

  function handleKeyDown(e) {
    if (e.key === "Escape") {
      setOpen(false);
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!open || suggestions.length === 0) return;
      e.preventDefault();
      const dir = e.key === "ArrowDown" ? 1 : -1;
      setActiveIndex((i) => (i + dir + suggestions.length) % suggestions.length);
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const picked = activeIndex >= 0 ? suggestions[activeIndex] : null;
      onSubmit(picked ? picked.name : value);
      setOpen(false);
    }
  }

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div className="flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 shadow-sm transition-colors focus-within:border-primary focus-within:ring-2 focus-within:ring-ring/40">
        <Search className="h-5 w-5 shrink-0 text-muted-foreground" aria-hidden="true" />
        <input
          type="text"
          value={value}
          role="combobox"
          aria-expanded={open}
          aria-controls="search-suggestions"
          aria-autocomplete="list"
          onChange={(e) => {
            onChange(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder="Search products..."
          className="w-full bg-transparent text-base outline-none placeholder:text-muted-foreground"
        />
        {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
        {value && !loading && (
          <button
            type="button"
            aria-label="Clear search"
            onClick={() => {
              onChange("");
              setOpen(false);
            }}
            className="rounded-md p-1 text-muted-foreground hover:bg-muted"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {open && suggestions.length > 0 && (
        <ul
          id="search-suggestions"
          role="listbox"
          className="absolute z-20 mt-2 w-full overflow-hidden rounded-xl border border-border bg-popover shadow-lg"
        >
          {suggestions.slice(0, 10).map((s, i) => (
            <li key={s.id ?? i} role="option" aria-selected={i === activeIndex}>
              <button
                type="button"
                onMouseEnter={() => setActiveIndex(i)}
                onClick={() => {
                  onSubmit(s.name);
                  setOpen(false);
                }}
                className={`flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left text-sm ${
                  i === activeIndex ? "bg-accent text-accent-foreground" : "text-foreground"
                }`}
              >
                <span className="truncate">{s.name}</span>
                <span className="shrink-0 text-xs text-muted-foreground">{s.brand}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
