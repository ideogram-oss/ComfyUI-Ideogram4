import { app } from "../../scripts/app.js";

// Credentials are stored ONLY in the node's gitignored ideogram_config.json (via
// the /ideogram/keys route). They are never kept as ComfyUI setting values, so
// they never reach comfy.settings.json or the GET /settings response.

const FIELDS = [
  {
    key: "IDEOGRAM_API_KEY",
    label: "Ideogram API Key",
    hint: "Magic Prompt provider: ideogram.",
  },
  {
    key: "OPENROUTER_API_KEY",
    label: "OpenRouter API Key",
    hint: "Magic Prompt provider: openrouter, with your chosen OpenRouter model.",
  },
  {
    key: "HF_TOKEN",
    label: "Hugging Face Token",
    hint:
      "Downloads the gated Ideogram weights, replacing `hf auth login`. You must " +
      "STILL accept the model's terms on its Hugging Face page once, in a browser. " +
      "Use a fine-grained, read-only token scoped to the Ideogram repos.",
  },
];

async function getStatus() {
  try {
    const r = await fetch("/ideogram/keys");
    return r.ok ? await r.json() : {};
  } catch {
    return {};
  }
}

async function postKeys(payload) {
  const r = await fetch("/ideogram/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`save failed (${r.status})`);
  return r.json();
}

function el(tag, style, props) {
  const e = Object.assign(document.createElement(tag), props || {});
  if (style) e.style.cssText = style;
  return e;
}

async function openDialog() {
  const status = await getStatus();

  const overlay = el(
    "div",
    "position:fixed;inset:0;z-index:10000;display:flex;align-items:center;" +
      "justify-content:center;background:rgba(0,0,0,0.5);"
  );
  const panel = el(
    "div",
    "background:var(--comfy-menu-bg,#202020);color:var(--fg-color,#fff);" +
      "border:1px solid var(--border-color,#444);border-radius:8px;padding:20px;" +
      "width:min(520px,92vw);box-shadow:0 8px 32px rgba(0,0,0,0.5);" +
      "font-family:sans-serif;"
  );
  panel.append(
    el("h3", "margin:0 0 14px;font-size:16px;", {
      textContent: "Ideogram 4.0 — API Keys",
    })
  );

  const close = () => overlay.remove();
  const inputs = {};

  for (const f of FIELDS) {
    const row = el("div", "margin-bottom:14px;");
    const top = el(
      "div",
      "display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;"
    );
    top.append(
      el("label", "font-size:13px;font-weight:600;", {
        textContent: `${f.label} (${f.key})`,
      })
    );
    const badge = el("span", "font-size:11px;opacity:0.75;", {
      textContent: status[f.key] ? "✓ Configured" : "Not set",
    });
    top.append(badge);
    row.append(top);

    const inputRow = el("div", "display:flex;gap:6px;align-items:center;");
    const input = el(
      "input",
      "flex:1;min-width:0;padding:6px 8px;border-radius:4px;" +
        "border:1px solid var(--border-color,#444);" +
        "background:var(--comfy-input-bg,#111);color:var(--input-text,#fff);",
      {
        type: "password",
        autocomplete: "off",
        placeholder: status[f.key]
          ? "Configured — leave blank to keep"
          : "Not set",
      }
    );
    inputs[f.key] = input;

    const clearBtn = el(
      "button",
      "padding:6px 10px;cursor:pointer;",
      { type: "button", textContent: "Clear" }
    );
    clearBtn.addEventListener("click", async () => {
      try {
        const s = await postKeys({ [f.key]: "" });
        input.value = "";
        input.placeholder = "Not set";
        badge.textContent = s[f.key] ? "✓ Configured" : "Cleared";
      } catch (e) {
        console.error("[Ideogram4]", e);
        badge.textContent = "Clear failed";
      }
    });

    inputRow.append(input, clearBtn);
    row.append(inputRow);
    row.append(
      el("div", "font-size:11px;opacity:0.65;margin-top:4px;", {
        textContent: f.hint,
      })
    );
    panel.append(row);
  }

  const footer = el(
    "div",
    "display:flex;justify-content:flex-end;gap:8px;margin-top:8px;"
  );
  const msg = el("span", "flex:1;font-size:12px;opacity:0.8;align-self:center;");
  const cancelBtn = el("button", "padding:6px 14px;cursor:pointer;", {
    type: "button",
    textContent: "Close",
  });
  cancelBtn.addEventListener("click", close);
  const saveBtn = el("button", "padding:6px 14px;cursor:pointer;font-weight:600;", {
    type: "button",
    textContent: "Save",
  });
  saveBtn.addEventListener("click", async () => {
    // Only send fields the user actually typed into; blank = keep existing.
    const payload = {};
    for (const f of FIELDS) {
      const v = inputs[f.key].value.trim();
      if (v) payload[f.key] = v;
    }
    if (!Object.keys(payload).length) {
      close();
      return;
    }
    msg.textContent = "Saving…";
    try {
      await postKeys(payload);
      close();
    } catch (e) {
      console.error("[Ideogram4]", e);
      msg.textContent = "Save failed";
    }
  });

  footer.append(msg, cancelBtn, saveBtn);
  panel.append(footer);

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  overlay.append(panel);
  document.body.append(overlay);
}

app.registerExtension({
  name: "Ideogram4.ApiKeys",
  settings: [
    {
      id: "Ideogram4.ApiKeysButton",
      name: "API Keys",
      category: ["Ideogram 4.0", "API Keys", "Manage"],
      type: () => {
        const btn = el("button", "padding:6px 14px;cursor:pointer;", {
          type: "button",
          textContent: "Manage API keys…",
        });
        btn.addEventListener("click", openDialog);
        return btn;
      },
      defaultValue: "",
    },
  ],
});
