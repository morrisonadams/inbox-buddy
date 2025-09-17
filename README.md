# Inbox Buddy (local, Dockerized)

A small local app that watches your Gmail, classifies new emails with Google Gemini, shows desktop notifications in your browser for important items, and lets you ask questions about recent emails.

## What you get

- FastAPI backend that polls Gmail and classifies with Gemini 1.5 Pro (override via `GOOGLE_GENAI_MODEL`)
- SQLite storage of parsed emails
- SSE stream that triggers browser notifications for important emails or replies needed
- Simple web UI with a chat style Q&A box
- Runs on your Mac with Docker Compose. You can also run the backend directly with Python if you prefer.

## Prereqs

1. **Docker Desktop for Mac** installed and running
2. **Google Gemini API key** from Google AI Studio
3. **Gmail API OAuth client** in Google Cloud Console

## Step 1. Create a Gemini API key

- Go to Google AI Studio and create an API key.
- Copy it. You will put it into `backend/.env` as `GOOGLE_GENAI_API_KEY`

## Step 2. Enable Gmail API and create OAuth client (Installed app)

1. Go to Google Cloud Console and create a project.
2. Enable the **Gmail API** for that project.
3. Configure the OAuth consent screen for "External" or "Internal" as needed.
4. Create **OAuth client ID** of type **Desktop app**.
5. Download the JSON and save it as `backend/credentials.json`.

> This app uses read only scope. You can change scopes in `gmail_client.py` if desired.

## Step 3. Configure env

Copy the sample env

```bash
cp backend/.env.sample backend/.env
# edit backend/.env and paste your GOOGLE_GENAI_API_KEY
```

If you prefer to run OAuth outside of Docker just once, you can run `python backend/app.py` locally, then copy the generated `token.json` into `backend/token.json`. The Docker container will pick it up.

## Step 4. Start the stack

```bash
docker compose up --build
```

- API: http://localhost:8000
- Web: http://localhost:5173

Open the web app. Click **Connect Gmail**. A browser window will pop for Google sign in. Approve it. On success a `token.json` file will be saved inside the container (and your mounted `backend/` folder if you run the API locally).

## How notifications work

- The web app subscribes to `/events` via SSE.
- When a new unread email arrives, the backend classifies it with Gemini.
- If it is important or reply needed, the page raises a browser notification. On macOS, Chrome and Safari will surface system notifications for localhost when permission is granted.

Click **Enable Notifications** in the UI and accept the browser prompt.

## Q&A over your inbox

Type questions like:
- "What did HR ask me to send this week?"
- "Summarize messages from Erin about the weekend"
- "Any invoices that need a reply?"

The backend sends a context window of recent emails to Gemini and returns the answer.

## Running outside Docker (optional)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
# add GOOGLE_GENAI_API_KEY to .env
python app.py
```

Then open http://localhost:8000/health to test, and point the frontend at `VITE_API_BASE=http://localhost:8000`.

## Notes and limits

- This is polling based. Default poll interval is 120 seconds. Adjust `POLL_INTERVAL` in `.env`.
- First run of OAuth uses a local HTTP callback on port 8081 inside the API container. Your browser will open. If you hit issues, do the OAuth step once outside Docker and copy `token.json` into `backend/`.
- Only read only Gmail scope is used. No send or modify actions are performed.
- The classifier is tuned to only flag reply-needed when someone is clearly waiting on you. Tweak thresholds in `.env` or refine the prompt in `triage.py`.
- Data stays local in `backend/db.sqlite3`.
- If you prefer Vertex AI instead of AI Studio keys you can swap the Gemini client code.

## Security

- Treat `token.json`, `credentials.json`, and your API key as secrets. Do not commit them.
- Scope is read only to reduce risk.
# inbox-buddy
