const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function jget(path){
  const r = await fetch(API + path);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
async function jpost(path, body){
  const r = await fetch(API + path, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadEmails(){
  const emails = await jget("/emails?limit=50&actionable_only=true");
  const el = document.getElementById("emails");
  el.innerHTML = "";
  if(emails.length === 0){
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "You're all caught up. No replies are needed right now.";
    el.appendChild(empty);
    return;
  }
  for(const e of emails){
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div><b>${e.subject || "(no subject)"}</b></div>
      <div>From: ${e.sender}</div>
      <div><i>${new Date(parseInt(e.internal_date || 0)).toLocaleString()}</i></div>
      <div>${e.snippet}</div>
      <div>Reply needed score: ${(e.reply_needed_score ?? 0).toFixed(2)} | Importance score: ${(e.importance_score ?? 0).toFixed(2)}</div>
    `;
    el.appendChild(card);
  }
}

async function ask(){
  const q = document.getElementById("question").value;
  if(!q) return;
  const res = await jpost("/ask",{question:q, limit:100});
  document.getElementById("answer").textContent = res.answer;
}

async function enableNotifs(){
  const perm = await Notification.requestPermission();
  if(perm !== "granted"){
    alert("Notifications not granted");
  }
}

function connectSSE(){
  const ev = new EventSource(API + "/events");
  ev.onmessage = (m)=>{
    try{
      const data = JSON.parse(m.data);
      if(data.type === "important_email" && data.actionable){
        if(Notification.permission === "granted"){
          new Notification("Reply needed", {
            body: data.subject + " — " + data.sender
          });
        }
      }
      if(data.type === "error"){
        console.warn("Server error:", data.message);
      }
    }catch(e){}
  };
}

document.getElementById("refresh").onclick = loadEmails;
document.getElementById("ask").onclick = ask;
document.getElementById("enableNotifs").onclick = enableNotifs;
document.getElementById("connect").onclick = () =>
  jget("/auth/start")
    .then(res => {
      if(res.already_authenticated){
        alert("Gmail is already connected.");
        return;
      }
      if(res.auth_url){
        window.open(res.auth_url, "_blank", "noopener,noreferrer");
        alert("Complete Google login in the opened tab, then return here.");
        return;
      }
      alert("Authentication response received.");
    })
    .catch(e => alert(e.message));

loadEmails();
connectSSE();
