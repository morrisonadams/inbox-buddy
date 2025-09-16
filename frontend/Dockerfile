FROM node:20-slim

WORKDIR /app
COPY package.json package-lock.json* /app/
RUN npm install --no-audit --no-fund
COPY . /app

ENV VITE_API_BASE=http://localhost:8000

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
