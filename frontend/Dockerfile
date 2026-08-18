# ==============================================================================
# Offline Intelligent Product Search — Frontend Dockerfile (Vite + React SPA)
# ==============================================================================

FROM node:20-alpine

WORKDIR /app

# Install dependencies first for optimal Docker layer caching
COPY package*.json ./
RUN npm install

# Copy application source code
COPY . .

# Build-time argument for backend API URL accessible from user browser
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

# Build production bundle
RUN npm run build

# Expose frontend application port
EXPOSE 3000

# Serve the production dist bundle on 0.0.0.0:3000
CMD ["npm", "run", "preview"]