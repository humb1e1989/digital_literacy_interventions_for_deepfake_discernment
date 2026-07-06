/**
 * Deployment configuration
 *
 * After deploying the backend to Railway:
 *   1. Replace the placeholder URL below with your actual Railway URL
 *   2. Commit and push — Vercel will redeploy automatically
 *
 * Local dev (python server.py on port 5001) works without changing anything.
 */

// Railway/Render backend URL — set this after deploying the backend
window.TRACKING_API  = 'http://localhost:5001';   // ← replace with https://your-app.railway.app
window.IMAGES_BASE   = 'http://localhost:5001';   // ← same URL (Railway serves images too)
