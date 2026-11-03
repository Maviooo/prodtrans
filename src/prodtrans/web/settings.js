"use strict";

// Widok ustawień: edytujemy kopię roboczą (draft), "Zapisz i zastosuj" wysyła ją na serwer.
(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_INPUTS_SHOWN = 64;
  const DEFAULT_VAD = {
    threshold: 0.5, min_silence_ms: 500, pre_roll_ms: 300,
    min_segment_s: 0.3, max_segment_s: 12.0, partial_every_s: 1.0,
  };
  // Suwaki: `toUi`/`fromUi` pozwalają pokazać próg jako "czułość" (odwrotnie do progu).
  const SLIDERS = [
    { key: "threshold", label: "Czułość wykrywania mowy", min: 0.1, max: 0.9, step: 0.05,
      toUi: (v) => +(1 - v).toFixed(2), fromUi: (v) => +(1 - v).toFixed(2), fmt: (v) => Math.round(v * 100) + "%",
      hint: "Wyżej: łapie cichsze głosy, ale też więcej szumu. Niżej: pomija szum, ale może zgubić ciche wypowiedzi." },
    { key: "min_silence_ms", label: "Cisza kończąca wypowiedź", min: 200, max: 1500, step: 50, fmt: (v) => v + " ms",
      hint: "Krócej: tekst pojawia się szybciej, ale zdania mogą się rwać na pauzach." },
    { key: "partial_every_s", label: "Odświeżanie tekstu na żywo", min: 0.5, max: 3, step: 0.25, fmt: (v) => v.toFixed(2).replace(".", ",") + " s",
      hint: "Jak często poprawia się tekst wstępny (kursywa) w trakcie mówienia." },
    { key: "max_segment_s", label: "Najdłuższy fragment", min: 5, max: 30, step: 1, fmt: (v) => v + " s",
      hint: "Dłuższa ciągła mowa jest dzielona na kawałki tej długości." },
    { key: "min_segment_s", label: "Najkrótsza wypowiedź", min: 0.1, max: 1, step: 0.05, fmt: (v) => v.toFixed(2).replace(".", ",") + " s",
      hint: "Krótsze dźwięki (trzaski, kaszel, stuknięcia) są pomijane." },
    { key: "pre_roll_ms", label: "Dźwięk przed mową", min: 0, max: 800, step: 50, fmt: (v) => v + " ms",
      hint: "Ile dźwięku sprzed wykrycia mowy dołączyć, żeby nie ucinać pierwszej sylaby." },
  ];

  let saved = null;
  let draft = null;
  let devices = [];
  let engines = [];
  let palette = [];
  let maxChannels = 16;
  let inputLevels = [];
  let loaded = false;

  const clone = (o) => JSON.parse(JSON.stringify(o));
  const isDirty = () => loaded && JSON.stringify(draft) !== JSON.stringify(saved);

  function plural(n, one, few, many) {
    if (n === 1) return one;
    const d = n % 10, t = n % 100;
    return d >= 2 && d <= 4 && (t < 12 || t > 14) ? few : many;
  }
  const inputsWord = (n) => `${n} ${plural(n, "wejście", "wejścia", "wejść")}`;
  const khz = (hz) => (hz / 1000).toLocaleString("pl-PL", { maximumFractionDigits: 1 }) + " kHz";

  function el(tag, props = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "style") Object.entries(v).forEach(([p, val]) => e.style.setProperty(p, val));
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k in e) e[k] = v;
      else e.setAttribute(k, v);
    }
    e.append(...children.filter((c) => c != null));
    return e;
  }

  function currentDevice() {
    return devices.find((d) => d.name === draft.device);
  }
  function inputCount() {
    const d = currentDevice();
    if (d) return Math.min(d.max_input_channels, MAX_INPUTS_SHOWN);
    return Math.max(1, ...draft.channels.map((c) => c.input + 1));
  }
  // Poziomy z serwera dotyczą urządzenia, które faktycznie działa (zapisanego).
  const levelsLive = () => PT.running && saved && draft.device === saved.device;

  // --- ładowanie ---
  async function loadAll() {
    const [s, d] = await Promise.all([fetch("/api/settings").then((r) => r.json()),
                                      fetch("/api/devices").then((r) => r.json())]);
    saved = s.settings;
    draft = clone(saved);
    engines = s.engines;
    palette = s.palette;
    maxChannels = s.max_channels;
    devices = d.devices;
    loaded = true;
    render();
  }

  function render() {
    renderDevice();
    renderInputs();
    renderChannels();
    renderEngines();
    renderVad();
    updateSaveBar();
  }

  // --- źródło dźwięku ---
  function renderDevice() {
    const sel = $("device");
    sel.replaceChildren();
    if (!devices.length) sel.append(el("option", { value: "", textContent: "Nie widzę żadnego urządzenia wejściowego" }));
    for (const d of devices) {
      sel.append(el("option", {
        value: d.name,
        textContent: `${d.name} · ${inputsWord(d.max_input_channels)} · ${khz(d.default_samplerate)}`,
      }));
    }
    if (draft.device && !currentDevice()) {
      sel.append(el("option", { value: draft.device, textContent: `${draft.device} (teraz niedostępne)` }));
    }
    sel.value = draft.device || "";

    const rate = $("rate");
    const dev = currentDevice();
    rate.replaceChildren(
      el("option", { value: "", textContent: "Automatycznie" + (dev ? ` (${khz(dev.default_samplerate)})` : "") }),
      ...[44100, 48000, 96000].map((hz) => el("option", { value: String(hz), textContent: khz(hz) })),
    );
    rate.value = draft.sample_rate ? String(draft.sample_rate) : "";
  }

  $("device").onchange = (e) => {
    draft.device = e.target.value || null;
    renderDevice();
    renderInputs();
    renderChannels();
    updateSaveBar();
  };
  $("rate").onchange = (e) => {
    draft.sample_rate = e.target.value ? Number(e.target.value) : null;
    updateSaveBar();
  };
  $("refresh").onclick = async () => {
    const btn = $("refresh");
    btn.disabled = true;
    btn.textContent = "Odświeżam…";
    try {
      devices = (await fetch("/api/devices?refresh=1").then((r) => r.json())).devices;
      render();
    } finally {
      btn.disabled = false;
      btn.textContent = "↻ Odśwież listę";
    }
  };

  function renderInputs() {
    const box = $("inputs");
    box.replaceChildren();
    const n = inputCount();
    for (let i = 0; i < n; i++) {
      const user = draft.channels.find((c) => c.input === i && c.enabled);
      const tile = el("div", { className: "in" + (user ? " used" : ""), title: user ? `Wejście ${i + 1} → ${user.name}` : `Wejście ${i + 1} (nieużywane)` },
        el("i"), el("b", { textContent: String(i + 1) }));
      if (user) tile.style.setProperty("--c", user.color);
      box.append(tile);
    }
    const hint = $("inputs-hint");
    if (!currentDevice()) hint.textContent = "Urządzenie jest niedostępne. Włącz je i kliknij „Odśwież listę”.";
    else if (!PT.running) hint.textContent = "Uruchom nasłuch, żeby zobaczyć, na których wejściach jest sygnał.";
    else if (!levelsLive()) hint.textContent = "Zapisz zmiany, żeby zobaczyć poziomy z nowego urządzenia.";
    else hint.textContent = "Mów do mikrofonu albo puść sygnał: kafelek z dźwiękiem się zapali.";
    paintLevels();
  }

  function paintLevels() {
    const live = levelsLive();
    const tiles = $("inputs").children;
    for (let i = 0; i < tiles.length; i++) {
      const db = live ? (inputLevels[i] ?? -90) : -90;
      const bar = tiles[i].firstChild;
      bar.style.height = PT.dbToPct(db) + "%";
      tiles[i].classList.toggle("hot", db > -6);
    }
    document.querySelectorAll(".chan-row").forEach((row) => {
      const c = draft.channels.find((x) => String(x.id) === row.dataset.id);
      const bar = row.querySelector(".meter > i");
      if (c && bar) bar.style.width = (live ? PT.dbToPct(inputLevels[c.input] ?? -90) : 0) + "%";
    });
  }

  PT.on("level", (ev) => {
    inputLevels = ev.inputs || [];
    if (!$("view-settings").hidden && loaded) paintLevels();
  });

  // --- kanały ---
  function renderChannels() {
    const list = $("chan-list");
    list.replaceChildren();
    const n = inputCount();
    const dev = currentDevice();
    draft.channels.forEach((c, idx) => {
      const bad = dev && c.input >= dev.max_input_channels;
      const row = el("div", { className: "chan-row" + (c.enabled ? "" : " off"), style: { "--c": bad ? "var(--err)" : c.color } });
      row.dataset.id = c.id;

      const color = el("input", { type: "color", value: c.color, title: "Kolor kanału", onchange: (e) => {
        c.color = e.target.value;
        row.style.setProperty("--c", c.color);
        swatch.style.setProperty("--c", c.color);
        renderInputs();
        updateSaveBar();
      } });
      const swatch = el("label", { className: "swatch", title: "Zmień kolor", style: { "--c": c.color } }, color);

      const name = el("input", { type: "text", value: c.name, maxLength: 40, placeholder: "Nazwa kanału", "aria-label": "Nazwa kanału",
        oninput: (e) => { c.name = e.target.value; updateSaveBar(); } });

      const select = el("select", { className: "in-select", "aria-label": "Wejście urządzenia",
        onchange: (e) => { c.input = Number(e.target.value); renderChannels(); renderInputs(); updateSaveBar(); } });
      for (let i = 0; i < n; i++) select.append(el("option", { value: String(i), textContent: `Wejście ${i + 1}` }));
      if (c.input >= n) select.append(el("option", { value: String(c.input), textContent: `Wejście ${c.input + 1} (brak)` }));
      select.value = String(c.input);
      if (bad) select.title = `To urządzenie ma ${inputsWord(dev.max_input_channels)}`;

      const meter = el("span", { className: "meter", title: "Poziom sygnału" }, el("i"));
      const enabled = el("label", { className: "switch", title: "Wyłączony kanał nie jest przepisywany" },
        el("input", { type: "checkbox", checked: c.enabled, onchange: (e) => {
          c.enabled = e.target.checked;
          row.classList.toggle("off", !c.enabled);
          renderInputs();
          updateSaveBar();
        } }), "Aktywny");
      const del = el("button", { className: "ghost icon", title: "Usuń kanał", "aria-label": `Usuń kanał ${c.name}`,
        disabled: draft.channels.length === 1, textContent: "✕",
        onclick: () => { draft.channels.splice(idx, 1); renderChannels(); renderInputs(); updateSaveBar(); } });

      row.append(swatch, name, select, meter, enabled, del);
      const twin = c.enabled && draft.channels.find((o) => o !== c && o.enabled && o.input === c.input);
      if (bad) {
        row.append(el("small", { className: "row-note bad", textContent:
          `To urządzenie ma ${inputsWord(dev.max_input_channels)}. Wybierz inne wejście.` }));
      } else if (twin) {
        row.append(el("small", { className: "row-note", textContent:
          `Wejście ${c.input + 1} słucha też kanał „${twin.name}”, więc wypowiedzi pojawią się dwa razy.` }));
      }
      list.append(row);
    });
    $("add-channel").disabled = draft.channels.length >= maxChannels;
    $("add-channel").textContent = draft.channels.length >= maxChannels
      ? `Osiągnięto limit ${maxChannels} kanałów` : "+ Dodaj kanał";
    paintLevels();
  }

  $("add-channel").onclick = () => {
    const id = Math.max(-1, ...draft.channels.map((c) => c.id)) + 1;
    const usedColors = new Set(draft.channels.map((c) => c.color));
    const usedInputs = new Set(draft.channels.map((c) => c.input));
    const n = inputCount();
    let input = 0;
    while (usedInputs.has(input) && input < n - 1) input++;
    draft.channels.push({
      id,
      name: `Kanał ${draft.channels.length + 1}`,
      color: palette.find((p) => !usedColors.has(p)) || palette[id % palette.length],
      input,
      enabled: true,
    });
    renderChannels();
    renderInputs();
    updateSaveBar();
    const rows = $("chan-list").querySelectorAll("input[type=text]");
    rows[rows.length - 1]?.select();
  };

  // --- silnik i zaawansowane ---
  function renderEngines() {
    const box = $("engines");
    box.replaceChildren();
    for (const e of engines) {
      const input = el("input", { type: "radio", name: "engine", value: e.id, checked: draft.engine === e.id,
        disabled: !e.available, onchange: () => { draft.engine = e.id; draft.model = null; updateSaveBar(); } });
      box.append(el("label", { className: "engine" + (e.available ? "" : " disabled") }, input,
        el("span", {}, el("b", { textContent: e.label }),
          el("small", { textContent: e.available ? e.hint : "Wymaga dodatku: uv sync --extra qwen" }))));
    }
    $("debug-record").checked = draft.debug_record;
    $("verify").checked = draft.verify;
    $("verify-lang").value = draft.verify_language;
    $("verify-options").hidden = !draft.verify;
  }
  $("verify").onchange = (e) => {
    draft.verify = e.target.checked;
    $("verify-options").hidden = !draft.verify;
    updateSaveBar();
  };

  $("verify-lang").onchange = (e) => { draft.verify_language = e.target.value; updateSaveBar(); };
  $("debug-record").onchange = (e) => { draft.debug_record = e.target.checked; updateSaveBar(); };

  function renderVad() {
    const box = $("vad");
    box.replaceChildren();
    for (const s of SLIDERS) {
      const toUi = s.toUi || ((v) => v);
      const fromUi = s.fromUi || ((v) => v);
      const id = "vad-" + s.key;
      const out = el("output", { htmlFor: id, textContent: s.fmt(toUi(draft.vad[s.key])) });
      const range = el("input", { type: "range", id, min: s.min, max: s.max, step: s.step, value: toUi(draft.vad[s.key]),
        oninput: (e) => {
          const v = Number(e.target.value);
          draft.vad[s.key] = s.key.endsWith("_ms") ? Math.round(fromUi(v)) : fromUi(v);
          out.textContent = s.fmt(v);
          updateSaveBar();
        } });
      box.append(el("div", { className: "slider" }, el("label", { htmlFor: id, textContent: s.label }), out, range,
        el("small", { textContent: s.hint })));
    }
  }
  $("vad-reset").onclick = () => { draft.vad = { ...DEFAULT_VAD }; renderVad(); updateSaveBar(); };

  // --- zapis ---
  function showError(msg) {
    const box = $("error");
    box.textContent = msg || "";
    box.hidden = !msg;
    if (msg) box.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function updateSaveBar() {
    $("savebar").hidden = !isDirty();
  }

  $("discard").onclick = () => { draft = clone(saved); showError(""); render(); };
  $("save").onclick = async () => {
    const btn = $("save");
    btn.disabled = true;
    btn.textContent = "Zapisuję…";
    try {
      const r = await fetch("/api/settings", {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(draft),
      });
      const body = await r.json();
      if (!r.ok) {
        showError(typeof body.detail === "string" ? body.detail : "Nie udało się zapisać ustawień.");
        return;
      }
      saved = body.settings;
      draft = clone(saved);
      showError("");
      render();
    } catch {
      showError("Brak połączenia z aplikacją. Sprawdź, czy ProdTrans działa.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Zapisz i zastosuj";
    }
  };

  window.addEventListener("beforeunload", (e) => { if (isDirty()) e.preventDefault(); });

  // Zmiany z innego okna albo start/stop: odśwież, jeśli użytkownik niczego nie edytuje.
  PT.on("hello", () => {
    if (!loaded) return;
    if (!isDirty()) loadAll(); else renderInputs();
  });
  PT.on("view", (view) => { if (view === "settings" && !loaded) loadAll(); });
  if (!$("view-settings").hidden) loadAll();
})();
