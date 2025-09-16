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
  const emails = await jget("/emails?limit=50");
  const el = document.getElementById("emails");
  el.innerHTML = "";
  for(const e of emails){
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = \`
      <div><b>\${e.subject || "(no subject)"}<\/b></div>
      <div>From: \${e.sender}</div>
      <div><i>\${new Date(parseInt(e.internal_date||0)).toLocaleString()}<\/i></div>
      <div>\${e.snippet}</div>
      <div>Important: \${e.is_important} | Reply needed: \${e.reply_needed}</div>
    \`;
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
      if(data.type === "important_email"){
        if(Notification.permission === "granted"){
          new Notification("Important email", {
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
document.getElementById("connect").onclick = ()=> jget("/auth/start").then(()=>alert("If prompted, complete Google login in the new tab.")).catch(e=>alert(e.message));

loadEmails();
connectSSE();
