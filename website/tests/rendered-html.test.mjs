import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the KuaiLive-M3 homepage", async () => {
  const response = await render("/");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /KuaiLive-M3/);
  assert.match(html, /Multi-Modal, Multi-Domain, and Multi-Feedback/);
  assert.match(html, /35M/);
  assert.match(html, /111M/);
  assert.match(html, /88M/);
  assert.match(html, /Cross-domain recommendation/);
  assert.match(html, /Live-stream highlight prediction/);
  assert.match(html, /Questionnaire-based recommendation/);
  assert.match(html, /Recall and NDCG @ 10, 20, 40/);
  assert.match(html, /Benchmark code/);
  assert.match(html, /github\.com\/imgkkk574\/KuaiLive-M3/);
  assert.match(html, /huggingface\.co\/datasets\/imgkkk2004\/KuaiLive-M3/);
  assert.doesNotMatch(html, /huggingface\.co\/imgkkk2004\/KuaiLive-M3/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/);
});

test("server-renders the field description page", async () => {
  const response = await render("/description/");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Field descriptions/);
  assert.match(html, /live_interaction\.csv/);
  assert.match(html, /live_emb_128_ts/);
  assert.match(html, /photo_play\.parquet/);
  assert.match(html, /huggingface\.co\/datasets\/imgkkk2004\/KuaiLive-M3/);
});

test("ships project metadata and social preview", async () => {
  const [layout, packageJson] = await Promise.all([
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  assert.match(layout, /KuaiLive-M3/);
  assert.match(layout, /og\.png/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await access(new URL("../public/og.png", import.meta.url));
});
