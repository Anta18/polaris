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
      
      console.log(`[postJSON] ${url} - Status: ${res.status} ${res.statusText}`);
      console.log(`[postJSON] Content-Type: ${res.headers.get("content-type")}`);
      console.log(`[postJSON] Content-Length: ${res.headers.get("content-length")}`);
      
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`POST ${url} ${res.status} ${res.statusText} ${txt}`);
      }
      
      // Check if response has content
      const contentType = res.headers.get("content-type");
      const contentLength = res.headers.get("content-length");
      
      if (contentLength === "0") {
        console.warn(`[postJSON] ${url} returned empty response (Content-Length: 0)`);
        return {} as T;
      }
      
      // Clone the response so we can read it multiple times if needed
      const responseText = await res.text();
      console.log(`[postJSON] Response body length: ${responseText.length}`);
      console.log(`[postJSON] Response body preview: ${responseText.substring(0, 200)}`);
      
      if (!responseText || responseText.trim() === "") {
        console.warn(`[postJSON] ${url} returned empty response body`);
        return {} as T;
      }
      
      try {
        return JSON.parse(responseText) as T;
      } catch (parseErr) {
        console.error(`[postJSON] Failed to parse JSON from ${url}:`, parseErr);
        console.error(`[postJSON] Response text:`, responseText);
        throw new Error(`Failed to parse JSON response from ${url}: ${parseErr}`);
      }
    } catch (err) {
      console.error(`[postJSON] Error on attempt ${attempt + 1}/${retries + 1} for ${url}:`, err);
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
