// Rwanda Tax AI Assistant - ChatGPT-style frontend (no frameworks)
// Connects to: http://127.0.0.1:8000/chat
//
// Request:  { message: "user text" }
// Response: { answer:  "bot response" }

// --- Configuration -----------------------------------------------------------
const API_URL = "http://127.0.0.1:8000/chat";

// --- UI state ----------------------------------------------------------------
// Simple in-memory chat history for this page session.
const state = {
  messages: [], // { role: "user" | "assistant", content: string, time: string }
};

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

function renderMessage(role, text, time) {
  const row = document.createElement("div");
  row.className = `bubble-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = time;

  const wrap = document.createElement("div");
  wrap.appendChild(bubble);
  wrap.appendChild(meta);

  row.appendChild(wrap);
  chatEl.appendChild(row);
  scrollToBottom();

  return { row, bubble };
}

function pushMessage(role, content) {
  const time = nowTime();
  state.messages.push({ role, content, time });
  return renderMessage(role, content, time);
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "bubble-row assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  bubble.innerHTML = `
    <div class="thinking">
      <span>typing...</span>
      <span class="dots">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </span>
    </div>
  `;

  row.appendChild(bubble);
  chatEl.appendChild(row);
  scrollToBottom();
  return row;
}

function autoResizeTextarea() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 170) + "px";
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function typeText(el, text, delayMs = 8) {
  el.textContent = "";
  for (let i = 0; i < text.length; i++) {
    el.textContent += text[i];
    if (i % 3 === 0) {
      scrollToBottom();
      const d = text[i] === " " ? Math.max(1, delayMs - 3) : delayMs;
      await sleep(d);
    }
  }
  scrollToBottom();
}

async function callChatApi(message) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }

  return await res.json();
}

async function sendMessage(message) {
  statusPill.textContent = "Thinking…";

  // 1) Show user bubble and store in UI state
  pushMessage("user", message);

  // 2) Show a typing indicator while waiting for the backend
  const typingRow = addTypingIndicator();

  try {
    // 3) Call backend
    const data = await callChatApi(message);
    const answer = data && typeof data.answer === "string" ? data.answer : "(No answer)";

    // 4) Replace typing indicator with the assistant answer
    typingRow.remove();
    const { bubble } = pushMessage("assistant", "");
    await typeText(bubble, answer, 6);
  } catch (err) {
    typingRow.remove();
    pushMessage("assistant", `Error: ${err.message}`);
  } finally {
    statusPill.textContent = "Ready";
  }
}

function resetChat() {
  state.messages = [];
  chatEl.innerHTML = `
    <div class="welcome">
      <div class="welcome-title">Ask Rwanda tax questions</div>
      <div class="welcome-text">
        This chatbot retrieves relevant passages from your indexed documents (FAISS) and uses an LLM to answer.
      </div>
    </div>
  `;
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

inputEl.addEventListener("keydown", (e) => {
  // Enter = send, Shift+Enter = newline
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

clearBtn.addEventListener("click", resetChat);

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

