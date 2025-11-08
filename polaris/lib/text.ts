// lib/text.ts
const REPLACEMENTS: [RegExp, string][] = [
  [/â€™/g, "’"],
  [/â€˜/g, "‘"],
  [/â€œ/g, "“"],
  [/â€?/g, "”"],
  [/â€“/g, "–"],
  [/â€”/g, "—"],
  [/â€¦/g, "…"],
  [/â€¢/g, "•"],
  [/â„¢/g, "™"],
  [/â/g, ""], // last-resort stray byte
];

export function cleanText(s: string | null | undefined) {
  if (!s) return s ?? "";
  let out = s;
  for (const [pat, rep] of REPLACEMENTS) out = out.replace(pat, rep);
  return out;
}
