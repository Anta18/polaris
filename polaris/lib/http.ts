// lib/http.ts
type PostOpts = {
  timeoutMs?: number;
  retries?: number;
  headers?: Record<string, string>;
  query?: Record<string, string | number | undefined>;
};

function withQuery(url: string, query?: PostOpts["query"]) {
  if (!query) return url;
  const u = new URL(url);
  for (const [k, v] of Object.entries(query)) {
    if (v !== undefined) u.searchParams.set(k, String(v));
  }
  return u.toString();
}

export async function postJSON<T>(
  url: string,
  body: unknown,
  opts: PostOpts = {}
): Promise<T> {
  const timeoutMs = opts.timeoutMs ?? 40_000;
  const retries = opts.retries ?? 1;
  const controller = new AbortController();

  const run = async (attempt: number): Promise<T> => {
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(withQuery(url, opts.query), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(opts.headers || {}),
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`POST ${url} ${res.status} ${res.statusText} ${txt}`);
      }
      return (await res.json()) as T;
    } catch (err) {
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 500 * Math.pow(2, attempt)));
        return run(attempt + 1);
      }
      throw err;
    } finally {
      clearTimeout(id);
    }
  };

  return run(0);
}
