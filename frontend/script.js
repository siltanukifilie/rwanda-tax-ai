// --- Configuration -----------------------------------------------------------
// If you run FastAPI on a different host/port, change it here.
const API_URL = "http://127.0.0.1:8000/chat";

// --- DOM elements ------------------------------------------------------------
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusPill = document.getElementById("statusPill");
const apiUrlLabel = document.getElementById("apiUrlLabel");

apiUrlLabel.textContent = API_URL;

// --- Helpers ----------------------------------------------------------------
function nowTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addBubble(role, text) {
  const row = document.createElement("div");
  row.className = `bubble-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = nowTime();

  const wrap = document.createElement("div");
  wrap.appendChild(bubble);
  wrap.appendChild(meta);

  row.appendChild(wrap);
  chatEl.appendChild(row);
  scrollToBottom();

  return { row, bubble };
}

function addThinkingBubble() {
  const row = document.createElement("div");
  row.className = "bubble-row assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";

  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.innerHTML = `
    <span>Thinking... 🤖</span>
    <span class="dots">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </span>
  `;

  bubble.appendChild(thinking);
  row.appendChild(bubble);
  chatEl.appendChild(row);
  scrollToBottom();

  return row;
}

async function sendMessage(message) {
  statusPill.textContent = "Thinking…";

  // Add user message
  addBubble("user", message);

  // Add thinking indicator
  const thinkingRow = addThinkingBubble();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }

    const data = await res.json();
    const answer = (data && data.answer) ? data.answer : "(No answer)";

    // Replace thinking indicator with assistant answer (typing effect)
    thinkingRow.remove();
    const { bubble } = addBubble("assistant", "");
    await typeText(bubble, answer, 6);
  } catch (err) {
    thinkingRow.remove();
    addBubble("assistant", `Error: ${err.message}`);
  } finally {
    statusPill.textContent = "Ready";
  }
}

function autoResizeTextarea() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 170) + "px";
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function typeText(el, text, delayMs = 8) {
  // Simple typing animation effect (hackathon-friendly)
  el.textContent = "";
  for (let i = 0; i < text.length; i++) {
    el.textContent += text[i];
    if (i % 3 === 0) {
      scrollToBottom();
      // Slightly faster on whitespace
      const d = text[i] === " " ? Math.max(1, delayMs - 3) : delayMs;
      await sleep(d);
    }
  }
  scrollToBottom();
}

// --- Events -----------------------------------------------------------------
sendBtn.addEventListener("click", async () => {
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = "";
  autoResizeTextarea();
  await sendMessage(message);
});

inputEl.addEventListener("input", autoResizeTextarea);

inputEl.addEventListener("keydown", async (e) => {
  // Enter = send, Shift+Enter = newline
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

clearBtn.addEventListener("click", () => {
  chatEl.innerHTML = `
    <div class="welcome">
      <div class="welcome-title">Ask Rwanda tax questions</div>
      <div class="welcome-text">
        This chatbot retrieves relevant passages from your indexed documents (FAISS) and uses an LLM to answer.
      </div>
    </div>
  `;
  scrollToBottom();
});

document.querySelectorAll("[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const example = btn.getAttribute("data-example");
    inputEl.value = example;
    autoResizeTextarea();
    inputEl.focus();
  });
});

// Initial UI setup
autoResizeTextarea();
scrollToBottom();

