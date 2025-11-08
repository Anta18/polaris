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
        <div className="flex items-center bg-zinc-900 rounded-lg shadow-lg overflow-hidden border-2 border-zinc-700 focus-within:border-blue-400 transition-colors">
          <Search className="w-5 h-5 text-gray-500 ml-5" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for news articles, topics, authors..."
            className="flex-1 px-4 py-4 text-base bg-transparent outline-none text-gray-100 placeholder:text-gray-500"
          />
          <button
            type="submit"
            className="p-[3px] relative group"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 rounded-lg" />
            <div className="px-10 py-4 bg-zinc-900 rounded-[6px] relative transition duration-200 text-white font-bold text-lg group-hover:bg-transparent">
              Search
            </div>
          </button>
        </div>
      </div>
    </form>
  );
}

