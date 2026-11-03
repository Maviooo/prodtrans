"use strict";

// Wspólne: połączenie WebSocket, statusy, Start/Stop, przełączanie widoków.
// settings.js nasłuchuje przez PT.on("level" | "hello", fn).
const PT = {
  handlers: {},
  running: false,
  on(type, fn) { (this.handlers[type] ||= []).push(fn); },
  emit(type, ev) { (this.handlers[type] || []).forEach((fn) => fn(ev)); },
};

const $ = (id) => document.getElementById(id);
const feed = $("feed");
const empty = $("empty");
const channelsNav = $("channels");
const jump = $("jump");
const toggle = $("toggle");

const channels = new Map(); // id -> {name, color, meter, card}
const entries = new Map(); // segment_id -> element
let hidden = new Set(load("hidden", []));
let fontPx = load("font", 20);
let stickToBottom = true;
let unseen = 0;

function load(key, fallback) {
  try { const v = localStorage.getItem("prodtrans." + key); return v ? JSON.parse(v) : fallback; }
  catch { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem("prodtrans." + key, JSON.stringify(value)); } catch { /* prywatne okno */ }
}

function setPill(el, text, kind) {
  el.textContent = text;
  el.title = text;
  el.className = "pill" + (kind ? " pill-" + kind : "");
}

// --- widoki ---
function route() {
  const view = location.hash.startsWith("#/ustawienia") ? "settings" : "live";
  $("view-live").hidden = view !== "live";
  $("view-settings").hidden = view !== "settings";
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  PT.emit("view", view);
}
window.addEventListener("hashchange", route);

// --- Start / Stop ---
function renderToggle() {
  toggle.disabled = false;
  toggle.textContent = PT.running ? "Zatrzymaj nasłuch" : "Start nasłuchu";
  toggle.className = "btn-run " + (PT.running ? "is-on" : "is-off");
}
toggle.onclick = async () => {
  toggle.disabled = true;
  try {
    const r = await fetch(PT.running ? "/api/stop" : "/api/start", { method: "POST" });
    PT.running = (await r.json()).running;
  } finally {
    renderToggle();
  }
};

// --- czcionka i czyszczenie ---
function applyFont() {
  document.documentElement.style.setProperty("--font", fontPx + "px");
  save("font", fontPx);
}
$("font-up").onclick = () => { fontPx = Math.min(48, fontPx + 2); applyFont(); };
$("font-down").onclick = () => { fontPx = Math.max(12, fontPx - 2); applyFont(); };
$("clear").onclick = () => {
  feed.querySelectorAll(".entry").forEach((e) => e.remove());
  entries.clear();
  empty.hidden = false;
};
applyFont();

// --- kanały na żywo ---
function buildChannels(list) {
  channels.clear();
  channelsNav.replaceChildren();
  for (const c of list) {
    const card = document.createElement("button");
    card.className = "chan" + (hidden.has(c.id) ? " off" : "");
    card.style.setProperty("--c", c.color);
    card.title = "Pokaż / ukryj kanał";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = c.name;
    const meter = document.createElement("span");
    meter.className = "meter";
    const bar = document.createElement("i");
    meter.append(bar);
    card.append(name, meter);
    card.onclick = () => toggleChannel(c.id);
    channelsNav.append(card);
    channels.set(c.id, { ...c, meter: bar, card });
  }
}

function toggleChannel(id) {
  if (hidden.has(id)) hidden.delete(id); else hidden.add(id);
  save("hidden", [...hidden]);
  channels.get(id)?.card.classList.toggle("off", hidden.has(id));
  for (const el of entries.values()) {
    el.classList.toggle("hidden", hidden.has(Number(el.dataset.channel)));
  }
}

function fmtTime(t) {
  return new Date(t * 1000).toLocaleTimeString("pl-PL", { hour12: false });
}

function scrollToEnd() {
  feed.scrollTop = feed.scrollHeight;
  unseen = 0;
  jump.hidden = true;
}
feed.addEventListener("scroll", () => {
  stickToBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60;
  if (stickToBottom) { unseen = 0; jump.hidden = true; }
});
jump.onclick = () => { stickToBottom = true; scrollToEnd(); };

function renderTranscript(ev) {
  const ch = channels.get(ev.channel) || { name: "Kanał " + (ev.channel + 1), color: "#999999" };
  let el = entries.get(ev.segment_id);
  if (ev.final && !ev.text) { // pusty finał: wypowiedzi nie było, usuń tekst wstępny
    if (el) { el.remove(); entries.delete(ev.segment_id); }
    empty.hidden = entries.size > 0;
    return;
  }
  const isNew = !el;
  if (isNew) {
    el = document.createElement("div");
    el.dataset.channel = ev.channel;
    const ts = document.createElement("span"); ts.className = "ts";
    const who = document.createElement("span"); who.className = "who";
    const txt = document.createElement("span"); txt.className = "txt";
    const text = document.createElement("span");
    const mark = document.createElement("span"); mark.className = "mark";
    txt.append(text, mark);
    el.append(ts, who, txt);
    feed.append(el);
    entries.set(ev.segment_id, el);
    empty.hidden = true;
  } else if (el.classList.contains("final") && !ev.final) {
    return; // spóźniony tekst wstępny
  }
  el.style.setProperty("--c", ch.color);
  el.className = "entry " + (ev.final ? "final" : "partial") + (hidden.has(ev.channel) ? " hidden" : "");
  el.children[0].textContent = fmtTime(ev.t_start);
  el.children[1].textContent = ch.name;
  const [text, mark] = el.children[2].children;
  text.textContent = ev.text;
  // Weryfikacja: "pending" = czeka na drugi model, "done" = sprawdzone (z poprawką albo bez).
  mark.className = "mark";
  mark.removeAttribute("title");
  if (ev.verify === "pending") mark.classList.add("pending");
  if (ev.verify === "done") {
    // Zmiana samej interpunkcji albo wielkości liter to nie poprawka, której trzeba się przyglądać.
    const norm = (t) => (t || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
    const changed = ev.draft && norm(ev.draft) !== norm(ev.text);
    mark.classList.add(changed ? "fixed" : "same");
    mark.title = changed ? `Przed weryfikacją: ${ev.draft}` : "Zweryfikowano, bez zmian";
  }

  if (stickToBottom) {
    scrollToEnd();
  } else if (isNew && !hidden.has(ev.channel)) {
    unseen += 1;
    jump.textContent = `Skocz do teraz (${unseen} nowych)`;
    jump.hidden = false;
  }
}

function dbToPct(db) {
  return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
}
PT.dbToPct = dbToPct;

function renderLevels(ev) {
  for (const [id, c] of channels) {
    c.meter.style.width = dbToPct(ev.channels?.[id] ?? -90) + "%";
  }
}

function renderStatus(st) {
  if (st.source === "engine" || st.source === "verifier") {
    const pill = $(st.source);
    const kind = { ok: "ok", loading: "warn", error: "err" }[st.state];
    setPill(pill, st.message, kind);
    pill.hidden = st.state === "ok"; // gotowy model nie potrzebuje miejsca w pasku
  } else {
    const kind = { ok: "ok", error: "err", stopped: "" }[st.state];
    setPill($("audio"), st.message, kind);
  }
}

function emptyMessage() {
  empty.replaceChildren();
  if (PT.running) {
    empty.textContent = "Czekam na mowę na kanałach…";
  } else {
    empty.append("Nasłuch jest zatrzymany. Kliknij „Start nasłuchu” albo przejdź do ");
    const a = document.createElement("a");
    a.href = "#/ustawienia";
    a.textContent = "ustawień";
    empty.append(a, ", żeby wybrać urządzenie i kanały.");
  }
}

function connect() {
  const conn = $("conn");
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { conn.hidden = true; };
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "hello") {
      PT.running = ev.running;
      renderToggle();
      buildChannels(ev.channels);
      feed.querySelectorAll(".entry").forEach((e) => e.remove());
      entries.clear();
      emptyMessage();
      empty.hidden = ev.history.length > 0;
      stickToBottom = true;
      ev.history.forEach(renderTranscript);
      ev.status.forEach(renderStatus);
      scrollToEnd();
      PT.emit("hello", ev);
    } else if (ev.type === "transcript") {
      renderTranscript(ev);
    } else if (ev.type === "level") {
      renderLevels(ev);
      PT.emit("level", ev);
    } else if (ev.type === "status") {
      renderStatus(ev);
    }
  };
  ws.onclose = () => {
    setPill(conn, "Brak połączenia z aplikacją, ponawiam…", "err");
    conn.hidden = false;
    setTimeout(connect, 1000);
  };
}

route();
connect();
