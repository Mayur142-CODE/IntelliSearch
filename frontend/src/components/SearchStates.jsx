import { AlertCircle, SearchX } from "lucide-react";

export function LoadingGrid() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="aspect-[4/3] bg-muted" />
          <div className="space-y-2 p-4">
            <div className="h-4 w-3/4 rounded bg-muted" />
            <div className="h-3 w-1/2 rounded bg-muted" />
            <div className="h-4 w-1/3 rounded bg-muted" />
          </div>
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ query }) {
  return (
    <div className="rounded-xl border border-dashed border-border bg-card px-6 py-16 text-center">
      <SearchX className="mx-auto h-8 w-8 text-muted-foreground" />
      <p className="mt-3 text-sm font-medium text-foreground">No products found for “{query}”</p>
      <p className="mt-1 text-sm text-muted-foreground">Try a different or shorter query.</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-6 py-12 text-center">
      <AlertCircle className="mx-auto h-8 w-8 text-destructive" />
      <p className="mt-3 text-sm font-medium text-foreground">Search request failed</p>
      <p className="mt-1 text-sm text-muted-foreground">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
        >
          Retry
        </button>
      )}
    </div>
  );
}
