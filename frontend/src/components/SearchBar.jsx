import { useEffect, useRef, useState } from "react";
import { Search, X, Loader2, Sparkles, Tag, DollarSign, Box, ArrowRight } from "lucide-react";

const TYPE_CONFIG = {
  brand:      { icon: Sparkles,   label: "Brand",      className: "text-violet-500" },
  category:   { icon: Tag,        label: "Category",   className: "text-blue-500" },
  product:    { icon: Box,        label: "Product",    className: "text-emerald-500" },
  correction: { icon: ArrowRight, label: "Did you mean", className: "text-amber-500" },
  price:      { icon: DollarSign, label: "Price",      className: "text-green-500" },
  phrase:     { icon: Search,     label: "Search",     className: "text-muted-foreground" },
};

function SuggestionIcon({ type }) {
  const config = TYPE_CONFIG[type] || TYPE_CONFIG.phrase;
  const Icon = config.icon;
  return <Icon className={`h-3.5 w-3.5 shrink-0 ${config.className}`} aria-hidden="true" />;
}

function SuggestionBadge({ type }) {
  const config = TYPE_CONFIG[type] || TYPE_CONFIG.phrase;
  return (
    <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium tracking-wide uppercase bg-muted/60 ${config.className}`}>
      {config.label}
    </span>
  );
}

export function SearchBar({
  value,
  onChange,
  onSubmit,
  suggestions = [],
  suggestionsLoading = false,
  open,
  setOpen,
}) {
  const [activeIndex, setActiveIndex] = useState(-1);
  const wrapperRef = useRef(null);
  const listRef = useRef(null);

  // Reset active index when suggestions change
  useEffect(() => setActiveIndex(-1), [suggestions]);

  // Scroll active item into view
  useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const items = listRef.current.querySelectorAll('[role="option"]');
      if (items[activeIndex]) {
        items[activeIndex].scrollIntoView({ block: "nearest" });
      }
    }
  }, [activeIndex]);

  // Click-outside handler
  useEffect(() => {
    function onClickOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false);
      }
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
      setActiveIndex((i) => {
        const next = i + dir;
        // Allow cycling and also allow -1 (no selection)
        if (next < -1) return suggestions.length - 1;
        if (next >= suggestions.length) return -1;
        return next;
      });
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      if (activeIndex >= 0 && suggestions[activeIndex]) {
        // Use the highlighted suggestion
        const picked = suggestions[activeIndex].text;
        onChange(picked);
        onSubmit(picked);
      } else {
        // Submit current input
        onSubmit(value);
      }
      setOpen(false);
    }
  }

  const hasSuggestions = suggestions.length > 0;
  const showDropdown = open && (hasSuggestions || suggestionsLoading) && value.trim().length > 0;

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div className="flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 shadow-sm transition-colors focus-within:border-primary focus-within:ring-2 focus-within:ring-ring/40">
        <Search className="h-5 w-5 shrink-0 text-muted-foreground" aria-hidden="true" />
        <input
          type="text"
          value={value}
          role="combobox"
          aria-expanded={showDropdown}
          aria-controls="autocomplete-listbox"
          aria-autocomplete="list"
          aria-activedescendant={activeIndex >= 0 ? `suggestion-${activeIndex}` : undefined}
          onChange={(e) => {
            onChange(e.target.value);
            setOpen(true);
          }}
          onFocus={() => {
            if (value.trim()) setOpen(true);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Search products — type and press Enter..."
          className="w-full bg-transparent text-base outline-none placeholder:text-muted-foreground"
        />
        {suggestionsLoading && (
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" aria-label="Loading suggestions" />
        )}
        {value && !suggestionsLoading && (
          <button
            type="button"
            aria-label="Clear search"
            onClick={() => {
              onChange("");
              onSubmit(""); // clear results too
              setOpen(false);
            }}
            className="rounded-md p-1 text-muted-foreground hover:bg-muted"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {showDropdown && (
        <ul
          ref={listRef}
          id="autocomplete-listbox"
          role="listbox"
          className="absolute z-30 mt-2 w-full overflow-y-auto rounded-xl border border-border bg-popover shadow-lg"
          style={{ maxHeight: "20rem" }}
        >
          {suggestionsLoading && suggestions.length === 0 && (
            <li className="flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>Finding suggestions...</span>
            </li>
          )}
          {suggestions.map((s, i) => (
            <li
              key={`${s.text}-${i}`}
              id={`suggestion-${i}`}
              role="option"
              aria-selected={i === activeIndex}
            >
              <button
                type="button"
                onMouseEnter={() => setActiveIndex(i)}
                onMouseLeave={() => setActiveIndex(-1)}
                onClick={() => {
                  onChange(s.text);
                  onSubmit(s.text);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors ${
                  i === activeIndex
                    ? "bg-accent text-accent-foreground"
                    : "text-foreground hover:bg-muted/50"
                }`}
              >
                <SuggestionIcon type={s.type} />
                <span className="flex-1 truncate">{s.text}</span>
                <SuggestionBadge type={s.type} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
