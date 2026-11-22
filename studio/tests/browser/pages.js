// Browser check of every studio page (jsdom): log in, open each page, run its
// scripts against a live server (fetch + WebSockets), start / stop each task.
// Prints one JSON line per page: JS errors and the resulting DOM state.
//
//   npm install jsdom@24 ws          (once, in this folder or with NODE_PATH)
//   node pages.js http://127.0.0.1:8890 <password>
//
// Run by ../test_pages_browser.py (skipped when node / jsdom are missing).
const { JSDOM, VirtualConsole, CookieJar } = require("jsdom");
const BASE = process.argv[2], PASSWORD = process.argv[3];
const jar = new CookieJar();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function nodeFetch(url, opts = {}) {
  const u = new URL(url, BASE).toString();
  const cookie = await jar.getCookieString(u);
  const r = await fetch(u, { ...opts, redirect: "manual", headers: { ...(opts.headers || {}), cookie } });
  for (const c of r.headers.getSetCookie ? r.headers.getSetCookie() : []) await jar.setCookie(c, u);
  return r;
}
async function login() {
  const r = await nodeFetch("/login");
  const html = await r.text();
  const xsrf = html.match(/name="_xsrf" value="([^"]+)"/)[1];
  const p = await nodeFetch("/login", { method: "POST", body: new URLSearchParams({ _xsrf: xsrf, password: PASSWORD, next: "/" }),
    headers: { "Content-Type": "application/x-www-form-urlencoded" } });
  if (p.status !== 302) throw new Error("login failed " + p.status);
}
async function open(path, action, waitMs = 3000) {
  const errors = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => errors.push("jsdomError: " + (e.message || e)));
  vc.on("error", (e) => errors.push("console.error: " + e));
  const dom = await JSDOM.fromURL(BASE + path, {
    cookieJar: jar, runScripts: "dangerously", resources: "usable", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) {
      w.fetch = (u, o) => nodeFetch(u, o);
      w.TextDecoder = TextDecoder;
      w.URL.createObjectURL = () => "blob:frame" + Math.random();
      w.URL.revokeObjectURL = () => {};
      w.confirm = () => true; w.alert = (m) => errors.push("alert: " + m);
      w.addEventListener("error", (e) => errors.push("window.error: " + e.message));
    },
  });
  const w = dom.window;
  await new Promise((r) => (w.document.readyState === "complete" ? r() : w.addEventListener("load", r)));
  // jsdom does not decode images: fire onload so the stream sends its acks
  let frames = 0;
  const imgTick = setInterval(() => {
    const v = w.document.getElementById("video");
    if (v && v.src && v.src !== v._seen) { v._seen = v.src; frames++; if (v.onload) v.onload(); }
  }, 30);
  await sleep(1500);
  let info = {};
  if (action) info = (await action(w)) || {};
  await sleep(waitMs);
  clearInterval(imgTick);
  info.frames = frames;
  const out = { path, errors, info };
  w.close();
  return out;
}
const txt = (w, id) => (w.document.getElementById(id) || {}).textContent;
(async () => {
  await login();
  const results = [];
  results.push(await open("/", async (w) => { await sleep(2500); return { res: txt(w, "res"), tasks: w.document.querySelectorAll("a.task").length, jobs: txt(w, "jobs").slice(0, 60), firstTask: w.document.querySelector("a.task h3").textContent }; }));
  for (const t of ["drive_check", "lane_follow", "navigate", "search"]) {
    results.push(await open("/task/" + t, async (w) => {
      const fields = w.document.querySelectorAll("#params .param").length;
      const startBtn = w.document.getElementById("start");
      startBtn.click();
      for (let i = 0; i < 100 && txt(w, "state") !== "執行中"; i++) await sleep(200);
      await sleep(2500);
      const state = txt(w, "state"), link = txt(w, "link"), tel = txt(w, "telemetry").slice(0, 80).replace(/\s+/g, " ");
      const extra = { drive_check: () => txt(w, "video-stats"), lane_follow: () => w.document.getElementById("lf-model").options.length + " models, stats: " + txt(w, "lf-stats").slice(0, 60),
        navigate: () => txt(w, "nv-labels") + " | " + txt(w, "nv-stats").slice(0, 50), search: () => txt(w, "sr-status") + " | targets: " + txt(w, "tg-list").slice(0, 40) }[t]();
      // live param change through the form (first live numeric field)
      w.document.getElementById("stop").click();
      for (let i = 0; i < 50 && txt(w, "state") === "執行中"; i++) await sleep(200);
      return { fields, state, link, tel, extra, after_stop: txt(w, "state"), events: w.document.querySelectorAll("#events li").length };
    }, 1000));
  }
  for (const p of ["/lane/datasets", "/lane/training", "/lane/models"]) {
    results.push(await open(p, async (w) => {
      await sleep(2500);
      if (p === "/lane/datasets") {
        // Open the first dataset when there is one
        const btn = w.document.querySelector("#ds-table button");
        if (btn) { btn.click(); await sleep(1500); }
        return { rows: w.document.querySelectorAll("#ds-table tr").length, thumbs: w.document.querySelectorAll("#ds-grid img").length, page: txt(w, "ds-page"), table: txt(w, "ds-table").slice(0, 60) };
      }
      if (p === "/lane/training") return { fields: w.document.querySelectorAll("#train-form .param").length, dataset_options: w.document.querySelector("#train-form select").options.length, jobs: txt(w, "train-jobs").slice(0, 40) };
      return { torch: w.document.querySelectorAll("#m-torch tr").length, trt: txt(w, "m-trt").slice(0, 30) };
    }, 500));
  }
  for (const r of results) console.log(JSON.stringify(r));
  process.exit(0);
})().catch((e) => { console.error("FATAL", e); process.exit(1); });
