"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const router = useRouter();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };

  return (
    <form onSubmit={handleSearch} className="w-full max-w-4xl mx-auto">
      <div className="relative">
        <div className="flex items-center bg-white dark:bg-slate-700 rounded-lg shadow-lg overflow-hidden border-2 border-slate-200 dark:border-slate-600 focus-within:border-blue-500 dark:focus-within:border-blue-500 transition-colors">
          <Search className="w-5 h-5 text-slate-400 ml-5" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for news articles, topics, authors..."
            className="flex-1 px-4 py-4 text-base bg-transparent outline-none text-slate-900 dark:text-white placeholder:text-slate-400"
          />
          <button
            type="submit"
            className="px-8 py-4 bg-blue-600 text-white font-semibold hover:bg-blue-700 transition-colors active:bg-blue-800"
          >
            Search
          </button>
        </div>
      </div>
    </form>
  );
}

