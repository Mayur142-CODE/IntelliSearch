import { createFileRoute } from "@tanstack/react-router";
// @ts-expect-error - JS component
import { SearchPage } from "../components/SearchPage.jsx";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "IntelliSearch — Offline Intelligent Product Search" },
      {
        name: "description",
        content:
          "Fuzzy, prefix and semantic product search UI with ranking score breakdown, powered by an offline search backend.",
      },
      { property: "og:title", content: "IntelliSearch — Offline Intelligent Product Search" },
      {
        property: "og:description",
        content: "Search products offline with fuzzy, prefix and semantic ranking scores.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: SearchPage,
});
