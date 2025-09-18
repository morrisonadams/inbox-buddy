const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function jget(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function jpost(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendChat");
const resetButton = document.getElementById("reset");

function addMessage(role, text, extras = {}) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  if (extras.error) {
    bubble.classList.add("error");
  }

  if (extras.title) {
    const title = document.createElement("div");
    title.className = "bubble-title";
    title.textContent = extras.title;
    bubble.appendChild(title);
  }

  if (extras.meta) {
    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.textContent = extras.meta;
    bubble.appendChild(meta);
  }

  if (text) {
    const body = document.createElement("div");
    body.className = "bubble-text";
    body.textContent = text;
    bubble.appendChild(body);
  }

  const summaryItems = Array.isArray(extras.summary)
    ? extras.summary
    : [];
  if (summaryItems.length) {
    const list = document.createElement("ul");
    list.className = "bubble-summary";
    for (const item of summaryItems) {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    }
    bubble.appendChild(list);
  }

  if (extras.reply) {
    const pre = document.createElement("pre");
    pre.className = "bubble-reply";
    pre.textContent = extras.reply;
    bubble.appendChild(pre);
  }

  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function normalizeSummary(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item || "").trim())
      .filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/\r?\n+/)
      .map((line) => line.replace(/^[\s*•-]+/, "").trim())
      .filter(Boolean);
  }
  return [];
}

async function loadEmails() {
  const el = document.getElementById("emails");
  el.innerHTML = "";
  let emails = [];
  try {
    emails = await jget("/emails?limit=50&actionable_only=true");
  } catch (err) {
    addMessage(
      "assistant",
      "I couldn't load your actionable emails. Please try refreshing.",
      { error: true }
    );
    return;
  }

  if (emails.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "You're all caught up. No replies are needed right now.";
    el.appendChild(empty);
    return;
  }

  for (const e of emails) {
    const card = document.createElement("div");
    card.className = "email-card";

    const subjectRow = document.createElement("div");
    const subjectStrong = document.createElement("strong");
    subjectStrong.textContent = e.subject || "(no subject)";
    subjectRow.appendChild(subjectStrong);
    card.appendChild(subjectRow);

    const senderRow = document.createElement("div");
    senderRow.textContent = `From: ${e.sender}`;
    card.appendChild(senderRow);

    const statusRow = document.createElement("div");
    statusRow.className = "email-status " + (e.is_unread ? "unread" : "read");
    statusRow.textContent = e.is_unread ? "Unread" : "Read";
    card.appendChild(statusRow);

    if (e.internal_date) {
      const dateRow = document.createElement("div");
      dateRow.className = "email-date";
      dateRow.textContent = new Date(Number(e.internal_date)).toLocaleString();
      card.appendChild(dateRow);
    }

    if (e.snippet) {
      const snippetRow = document.createElement("div");
      snippetRow.textContent = e.snippet;
      card.appendChild(snippetRow);
    }

    const scoresRow = document.createElement("div");
    scoresRow.className = "email-scores";
    const replyScore = Number(e.reply_needed_score ?? 0).toFixed(2);
    const importanceScore = Number(e.importance_score ?? 0).toFixed(2);
    scoresRow.textContent = `Reply score ${replyScore} · Importance ${importanceScore}`;
    card.appendChild(scoresRow);

    if (e.assistant_message) {
      const note = document.createElement("div");
      note.className = "assistant-note";
      note.textContent = e.assistant_message;
      card.appendChild(note);
    }

    const summary = normalizeSummary(e.assistant_summary);
    if (summary.length) {
      const list = document.createElement("ul");
      list.className = "assistant-summary";
      for (const item of summary) {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
      }
      card.appendChild(list);
    }

    if (e.assistant_reply) {
      const pre = document.createElement("pre");
      pre.className = "assistant-reply";
      pre.textContent = e.assistant_reply;
      card.appendChild(pre);
    }

    el.appendChild(card);
  }
}

let sending = false;
let resetting = false;
let awaitingResetAck = false;

async function sendChat() {
  if (sending) return;
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage("user", message);
  chatInput.value = "";
  sending = true;
  sendButton.disabled = true;

  try {
    const res = await jpost("/ask", { question: message, limit: 100 });
    const answer = (res.answer || "I couldn't find anything helpful just yet.").trim();
    addMessage("assistant", answer);
  } catch (err) {
    console.error(err);
    addMessage(
      "assistant",
      "Sorry, I ran into an error while checking your inbox: " + err.message,
      { error: true }
    );
  } finally {
    sending = false;
    sendButton.disabled = false;
    chatInput.focus();
  }
}

async function resetInbox() {
  if (resetting) return;
  const confirmed = window.confirm(
    "Resetting clears stored emails so they can be reanalyzed. Continue?"
  );
  if (!confirmed) return;

  resetting = true;
  resetButton.disabled = true;
  let requestSucceeded = false;
  try {
    awaitingResetAck = true;
    const res = await jpost("/reset", {});
    requestSucceeded = true;
    const deleted = Number(res.deleted ?? 0);
    const message = deleted
      ? `Cleared ${deleted} stored email${deleted === 1 ? "" : "s"}. I'll reanalyze now.`
      : "I cleared the stored emails. I'll reanalyze your inbox now.";
    addMessage("assistant", message);
    loadEmails();
  } catch (err) {
    awaitingResetAck = false;
    addMessage(
      "assistant",
      "Sorry, I couldn't reset the inbox: " + err.message,
      { error: true }
    );
  } finally {
    resetting = false;
    resetButton.disabled = false;
    if (!requestSucceeded) {
      awaitingResetAck = false;
    }
  }
}

async function enableNotifs() {
  if (!("Notification" in window)) {
    alert("Notifications are not supported in this browser.");
    return;
  }
  const perm = await Notification.requestPermission();
  if (perm !== "granted") {
    alert("Notifications not granted");
  }
}

function addAssistantAlert(data) {
  const summary = normalizeSummary(data.assistant_summary);
  const replyDraft = typeof data.assistant_reply === "string"
    ? data.assistant_reply.trim()
    : "";
  const message = (data.assistant_message || "").trim() ||
    `You have an actionable email from ${data.sender || "someone"}.`;

  const metaParts = [];
  if (data.subject) metaParts.push(`Subject: ${data.subject}`);
  if (data.reply_needed_score !== undefined && data.importance_score !== undefined) {
    metaParts.push(
      `Scores — reply ${(Number(data.reply_needed_score) || 0).toFixed(2)}, ` +
      `importance ${(Number(data.importance_score) || 0).toFixed(2)}`
    );
  }

  addMessage("assistant", message, {
    title: `New email from ${data.sender || "Someone"}`,
    meta: metaParts.join(" • ") || undefined,
    summary,
    reply: replyDraft,
  });
}

function connectSSE() {
  let announced = false;
  let errorNotified = false;
  const source = new EventSource(API + "/events");

  source.onopen = () => {
    errorNotified = false;
    if (!announced) {
      addMessage(
        "assistant",
        "I'm connected to Gmail updates. I'll ping you when something needs attention."
      );
      announced = true;
    }
  };

  source.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "important_email" && data.actionable) {
        addAssistantAlert(data);
        if ("Notification" in window && Notification.permission === "granted") {
          const body = (data.assistant_message || `${data.subject || "(no subject)"} — ${data.sender || ""}`)
            .slice(0, 140);
          new Notification("Reply needed", {
            body,
          });
        }
        loadEmails();
      } else if (data.type === "auth_required") {
        addMessage(
          "assistant",
          "Please connect Gmail so I can keep watching for new messages.",
          { error: true }
        );
      } else if (data.type === "error") {
        addMessage(
          "assistant",
          `I ran into an error while polling Gmail: ${data.message}`,
          { error: true }
        );
      } else if (data.type === "reset") {
        const deleted = Number(data.deleted ?? 0);
        if (awaitingResetAck) {
          awaitingResetAck = false;
        } else {
          const message = deleted
            ? `Stored email history was cleared (${deleted} message${deleted === 1 ? "" : "s"}).`
            : "Stored email history was cleared.";
          addMessage("assistant", message);
        }
        loadEmails();
      }
    } catch (err) {
      console.warn("Failed to parse SSE payload", err);
    }
  };

  source.onerror = () => {
    if (!errorNotified) {
      addMessage(
        "assistant",
        "I lost connection to Gmail updates. I'll retry automatically.",
        { error: true }
      );
      errorNotified = true;
    }
  };

  return source;
}

sendButton.onclick = sendChat;
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChat();
  }
});

document.getElementById("refresh").onclick = loadEmails;
document.getElementById("enableNotifs").onclick = enableNotifs;
document.getElementById("connect").onclick = () =>
  jget("/auth/start")
    .then((res) => {
      if (res.already_authenticated) {
        alert("Gmail is already connected.");
        return;
      }
      if (res.auth_url) {
        window.open(res.auth_url, "_blank", "noopener,noreferrer");
        alert("Complete Google login in the opened tab, then return here.");
        return;
      }
      alert("Authentication response received.");
    })
    .catch((e) => alert(e.message));
resetButton.onclick = resetInbox;

addMessage(
  "assistant",
  "Hi! I'm Inbox Buddy. Ask me about your inbox or wait for me to nudge you when a reply is needed."
);

loadEmails();
connectSSE();
